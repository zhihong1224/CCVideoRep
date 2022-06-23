# %%writefile pretrain.py
import argparse
import os, math, random, time, gc, sys, json, psutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import logging
# from imp import reload

# reload(logging)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt='%H:%M:%S',
#     handlers=[
#         logging.FileHandler(f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
#         logging.StreamHandler()
#     ]
# )

import numpy as np
import pandas as pd

from config.data_cfg import *
from config.model_cfg import *
from config.pretrain_cfg import *
from data.record_trans import record_transform
from data.qq_dataset import QQDataset
from qqmodel.qq_uni_model import QQUniModel
from optim.create_optimizer import create_optimizer
from utils.eval_spearman import evaluate_emb_spearman
from utils.utils import set_random_seed

from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset
from qqmodel.model_utils import AverageMeter, setup_logger

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ChainDataset
import torch.distributed as dist
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

gc.enable()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed(SEED)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_pred_and_loss(model, item, task=None):
    """Get pred and loss for specific task"""
    video_feature = item['frame_features'].to(DEVICE)
    input_ids = item['id'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)

    target = None
    if 'target' in item:
        target = item['target'].to(DEVICE)

    pred, emb, loss = model(video_feature, video_mask, input_ids, attention_mask, target, task)
    return pred, emb, loss


def main():
#     os.environ['CUDA_VISIBLE_DEVICES']='0'
    # todo: add args for ddp train---
    parser = argparse.ArgumentParser(description="Pretrain")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--distributed", type=bool, default=False)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
        
    logger = setup_logger("ddp_train", "log", get_rank(), "bs128.txt")

    # Show config
    logger.info("Start")
    for fname in ['pretrain', 'model', 'data']:
        logger.info('=' * 66)
        with open(f'config/{fname}_cfg.py') as f:
            logger.info(f"Config - {fname}:" + '\n' + f.read().strip())

    list_val_loss = []
    logger.info(f"Model_type = {MODEL_TYPE}")
    trans = record_transform(model_path=BERT_PATH,
                             tag_file=f'{DATA_PATH}/tag_list.txt',
                             get_tagid=True)

    for fold in range(NUM_FOLDS):
        logger.info('=' * 66)
        model_path = "checkpoints"
        logger.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={SEED + fold}")

        set_random_seed(SEED + fold)

        val_loss = train(args.local_rank, args.distributed, trans, logger, model_path)

        list_val_loss.append(val_loss)

        gc.collect()

        logger.info(f"Fold{fold} val_loss_list=" + str([round(kk, 6) for kk in list_val_loss]))

    logger.info(f"Val Cv={np.mean(list_val_loss):6.4} +- {np.std(list_val_loss):6.4}")
    logger.info("Train finish")


def train(local_rank, distributed, trans, logger, model_path):
    # data
    train_dataset = QQDataset([f"{DATA_PATH}/pairwise/pairwise.tfrecords"], trans, desc=DESC)
    # todo: 注 在record_trans.parse_tfrecord中拿到title, 'text_id' [CLS][SEP] text [SEP]
    # todo: 'text_mask' [1,1,1,...,0,0], 'frame_features': np array (1536,)
    # todo: 'tag_id' [81774,...]
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)
        
    val_loader = train_loader
    total_steps = NUM_EPOCHS * len(train_dataset) // BATCH_SIZE // get_world_size()
    warmup_steps = int(WARMUP_RATIO * total_steps)
    logger.info(f'Total train steps={total_steps}, warmup steps={warmup_steps}')

    # model
    model = QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)
    model.to(DEVICE)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
    # optimizer
    optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # schedueler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps,
                                                num_warmup_steps=warmup_steps)

    save_to_disk = get_rank() == 0

    # train model---
    best_val_loss, best_epoch, step = None, 0, 0
    train_loss = AverageMeter("Loss", ':.4e')

    for epoch in range(NUM_EPOCHS):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        for batch_num, item in enumerate(train_loader):
            model.train()
            pred, emb, loss = get_pred_and_loss(model, item)  # pred:(bn,categories) emb:(bn,256) loss
            torch.distributed.barrier()
            if distributed:
                reduced_loss = reduce_mean(loss, get_world_size())
            else:
                reduced_loss = loss
            train_loss.update(reduced_loss.item(), pred.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            logger.info(
                f"Epoch={epoch + 1}/{NUM_EPOCHS}|step={step:3}|train_loss={train_loss.avg:6.4})")

            if (step + 1) % 50 == 0:
                val_loss, emb, vid_l = eval(distributed, model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=10000)
                
#                 label, spear = evaluate_emb_spearman(emb, vid_l, label_path=f"{DATA_PATH}/pairwise/label.tsv")
                logger.info(
                    f"Epoch={epoch + 1}/{NUM_EPOCHS}|step={step:3}|val_loss={val_loss:6.4}")
                
                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_to_disk:  # todo: 主进程保存模型
                        torch.save(model.state_dict(), os.path.join(model_path, 'best.pth'))

            step += 1
    
        if save_to_disk:
            torch.save({
                'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(model_path, 'model_'+str(epoch)+'.pth'))
            
    return best_val_loss


def eval(distributed, model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):
    """Evaluates the |model| on |data_loader|"""
    torch.cuda.empty_cache()
    model.eval()
    loss_l, emb_l, vid_l = [], [], []

    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, emb, loss = get_pred_and_loss(model, item, task='tag')

            if loss is not None:
                if distributed:
                    reduced_loss = reduce_mean(loss, get_world_size())
                else:
                    reduced_loss = loss
                loss_l.append(reduced_loss.to("cpu"))

            emb_l += emb.to("cpu").tolist()

            vid_l.append(item['vid'][0].numpy())

            if (batch_num + 1) * emb.shape[0] >= eval_max_num:
                break
    synchronize()
    torch.cuda.empty_cache()
    return np.mean(loss_l), np.array(emb_l), np.concatenate(vid_l)


if __name__ == '__main__':
    main()
