# Test
import torch
import time
from baseline import Model
from common.config import parse_opts
from common.logger import Logger, AverageMeter
from data.dataset import FSSDataset
from common import utils
from common.evaluation import Evaluator
import torch.optim as optim
from torch import nn
import os
import datetime
import cv2
import neptune

os.environ['CUDA_VISIBLE_DEVICES']= '2, 3, 4, 5, 6, 7'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "DETAIL"
local_rank = int(os.environ["LOCAL_RANK"])
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYWYzMTAyMy1jMzkyLTQzZGYtOThiMC0xZWIxMmZhODU3OTIifQ=="



def main_process(local_rank):
    return not args.distributed or (args.distributed and (local_rank == 0))


def train(epoch, model, dataloader, optimizer, cross_entropy_loss, training):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    if args.distributed:
        if training:
            model.module.train()
            model.module.feat_extractor.eval()
        else:
            model.module.eval()
    else:
        model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        
        # 1. forward pass
        batch = utils.to_cuda(batch)

        if training:
            logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), 1)
        else:
            logit_mask = model.module(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), 1)
        pred_mask = logit_mask.argmax(dim=1)
       # 2. Compute loss & update model parameters
        # loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = batch['query_mask'].view(bsz, -1).long()
        loss = cross_entropy_loss(logit_mask, gt_mask)
        

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        if main_process(local_rank):
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=20)
            # if training:
                # average_meter.loss_plot(idx, plot_idx=76)
    

    # Write evaluation results
    if main_process(local_rank):
        average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


def main():
    global args
    args = parse_opts()
    # model initialization
    if args.backbone == 'swin-b':
        _feat_channels = [128, 256, 512, 1024]
        _conv_dim = 256
        _mask_dim = 256
    if args.backbone == 'swin-l':
        _feat_channels = [192, 384, 768, 1536]
        _conv_dim = 384
        _mask_dim = 384
    if args.backbone == 'resnet101' or args.backbone == 'resnet50':
        _feat_channels = [256, 512, 1024, 2048]
    """
    model = Model(backbone=args.backbone, pretrained_path=args.feature_extractor_path, use_original_imgsize=False,
                  feat_channels=_feat_channels, n_layers=[2, 18, 2], conv_dim=_conv_dim, mask_dim=_mask_dim)
    """
    model = Model(backbone=args.backbone, pretrained_path=args.feature_extractor_path, use_original_imgsize=False)
    # optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
    #                         "momentum": 0.9, "weight_decay": args.lr / 10, "nesterov": True}])
    # optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr, "weight_decay": args.lr/100}])
    optimizer = optim.AdamW([{"params": model.parameters(), "lr": args.lr, "weight_decay": 0.05}])
    cross_entropy_loss = nn.CrossEntropyLoss()
    Evaluator.initialize()
    # 训练数据集
    FSSDataset.initialize(img_size=384, use_original_imgsize=False)
    # 测试数据集

    if torch.cuda.device_count() > 1:
        args.distributed = True
    else:
        args.distributed = False

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=600))
        # local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        # global device
        device = torch.device("cuda", local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        # model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if args.resume != '':
            params = model.state_dict()
            state_dict = torch.load(args.resume, map_location=torch.device('cpu'))
            for k1, k2 in zip(list(state_dict.keys()), params.keys()):
                state_dict[k2] = state_dict.pop(k1)
            model.load_state_dict(state_dict)
        if local_rank == 0:
            for name, param in model.named_parameters():
                if param.requires_grad is True:
                    print(name)
                else:
                    print("no grad", name)
            Logger.initialize(args, training = True)
            Logger.info('# available GPUs: %d' % torch.cuda.device_count())
            # model_version = neptune.init_model_version(project="wujr/Model", model=args.model_name, api_token=API_TOKEN)
            # model_version["model/parameters"] = args
            # run = neptune.init_run(project="wujr/Model", api_token=API_TOKEN, source_files=["**/*.py", "**/*.sh"])
            # run["parameters"] = args
            # run["sys/tags"].add("train")
            # run["sys/tags"].add(args.benchmark)
            # model_version["run/id"] = run["sys/id"].fetch()
            # model_version["sys/tags"].add(args.benchmark)
            # if args.add_mask:
                # model_version["sys/tags"].add("mask")
            # else:
                # model_version["sys/tags"].add("no mask")
            # model_version["sys/tags"].add(str(args.num_queries) + " queries" )
    else:
        device = torch.device("cuda")
        model.to(device)
        model.train()
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())
        dataloader_val = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, args.test_dataset, args.bsz, args.nworker, args.fold, 'val', training=False)
    
    dataloader_trn = FSSDataset.build_dataloader(args.datapath, args.test_num, args.distributed, args.benchmark, args.bsz, args.nworker, args.fold,
                                                 'trn', training=True)    
    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.nepoch):
        if args.distributed:
            dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, cross_entropy_loss, training=True)
        # if main_process(local_rank):
            # run["train/loss"].append(trn_loss)
            # run["train/miou"].append(trn_miou)
        # evaluation
        if main_process(local_rank):
             with torch.no_grad():
                dataloader_val = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, args.test_dataset, args.bsz, 0, args.fold, 'val', training=False)
                
                val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, cross_entropy_loss, training=False)
                # _, animal_val_miou, _ = train(epoch, model, dataloader_val_animal, optimizer, cross_entropy_loss, training=False)
                # _, eyeballs_val_miou, _ = train(epoch, model, dataloader_val_eyeballs, optimizer, cross_entropy_loss, training=False)
                # _, aerial_val_miou, _ = train(epoch, model, dataloader_val_aerial, optimizer, cross_entropy_loss, training=False)
                # run["validation/miou_animal"].append(animal_val_miou)
                # run["validation/miou_eyeballs"].append(eyeballs_val_miou)
                # run["validation/miou_aerial"].append(aerial_val_miou)

                # Save the best model
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    if main_process(local_rank):
                        # run["validation/best_miou_animal"] = best_animal
                        # run["validation/best_miou_eyeballs"] = best_eyeballs
                        # run["validation/best_miou_aerial"] = best_aerial
                        # model_version["validation/best_miou"] = best_val_miou
                        model_version = None
                        run = None
                        Logger.save_model_miou(model, epoch, best_val_miou, run, model_version)
                                    
  


if __name__=="__main__":
    main()
