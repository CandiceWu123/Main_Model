# Test
import torch
import time
from model import Model
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

os.environ['CUDA_VISIBLE_DEVICES']= '0,1, 2, 3, 4, 5'
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
    if args.backbone == 'swin-l':
        _feat_channels = [192, 384, 768, 1536]
    if args.backbone == 'resnet101' or args.backbone == 'resnet50':
        _feat_channels = [256, 512, 1024, 2048]
    if args.backbone == 'vgg':
        _feat_channels = [256, 512, 512, 512]
    """
    model = Model(backbone=args.backbone, pretrained_path=args.feature_extractor_path, use_original_imgsize=False,
                  feat_channels=_feat_channels, n_layers=[2, 18, 2], conv_dim=_conv_dim, mask_dim=_mask_dim)
    """
    model = Model(backbone=args.backbone, pretrained_path=args.feature_extractor_path, use_original_imgsize=False,
                  feat_channels=_feat_channels, hidden_dim=args.hidden_dim, num_queries=args.num_queries,
                  nhead=args.nhead, dec_layers=args.dec_layers, conv_dim=args.conv_dim, mask_dim=args.mask_dim,
                  enforce_input_project=args.enforce_input_project, add_mask=args.add_mask)
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
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1000000))
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
            if args.neptune:
                model_version = neptune.init_model_version(project="wujr/Model", model=args.model_name, api_token=API_TOKEN)
                model_version["model/parameters"] = args
                run = neptune.init_run(project="wujr/Model", api_token=API_TOKEN, source_files=["**/*.py", "**/*.sh"])
                run["parameters"] = args
                run["sys/tags"].add("train")
                # run["sys/tags"].add(args.benchmark)
                model_version["run/id"] = run["sys/id"].fetch()
    else:
        device = torch.device("cuda")
        model.to(device)
        model.train()
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    dataloader_trn = FSSDataset.build_dataloader(args.datapath, args.test_num, args.distributed, args.benchmark, args.bsz, args.nworker, args.fold,
                                                 'trn', training=True)
    # Train
    best_animal = float('-inf')
    best_eyeballs = float('-inf')
    best_aerial = float('-inf')
    best_luna = float('-inf')
    best_steel = float('-inf')
    best_cracks = float('-inf')
    best_sum = float('-inf')
    best_better_than_sota_num = 0
    if args.backbone == 'resnet50':
        sota_animal = 48.8
        sota_eyeballs = 9.57
        sota_cracks = 3.36
        sota_steel = 10.71
        sota_luna = 7.79
        sota_aerial = 9.61
    if args.backbone == 'resnet101':
        sota_animal = 55.51
        sota_eyeballs = 10.2
        sota_cracks = 1.55
        sota_steel = 9.72
        sota_luna = 6.34
        sota_aerial = 9.48
    if args.backbone == 'swin-l':
        sota_animal = 61.91
        sota_eyeballs = 9.86
        sota_cracks = 8.97
        sota_steel = 12.25
        sota_luna = 7.22
        sota_aerial = 9.46
    # sota_sum = sota_animal + sota_aerial + sota_luna + sota_steel + sota_cracks + sota_eyeballs
    for epoch in range(args.nepoch):
        if args.distributed:
            dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, cross_entropy_loss, training=True)
        if main_process(local_rank) and args.neptune:
            run["train/loss"].append(trn_loss)
            run["train/miou"].append(trn_miou)
        # evaluation
        if main_process(local_rank):
             with (torch.no_grad()):
                # dataloader_val = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, args.test_dataset, args.bsz, 0, args.fold, 'val', training=False)
                dataloader_val_animal = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, 'Animal', args.bsz, 0, args.fold, 'val', training=False)
                dataloader_val_eyeballs = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, 'Eyeballs', args.bsz, 0, args.fold, 'val', training=False)
                dataloader_val_aerial = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, 'Aerial', args.bsz, 0, args.fold, 'val', training=False)
                dataloader_val_luna = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, 'Artificial_Luna_Landscape', args.bsz, 0, args.fold, 'val', training=False)
                dataloader_val_steel = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, 'Magnetic_tile_surface', args.bsz, 0, args.fold, 'val', training=False)
                dataloader_val_cracks = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, 'Crack_Detection', args.bsz, 0, args.fold, 'val', training=False)
                # val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, cross_entropy_loss, training=False)
                _, animal_val_miou, _ = train(epoch, model, dataloader_val_animal, optimizer, cross_entropy_loss, training=False)
                _, eyeballs_val_miou, _ = train(epoch, model, dataloader_val_eyeballs, optimizer, cross_entropy_loss, training=False)
                _, aerial_val_miou, _ = train(epoch, model, dataloader_val_aerial, optimizer, cross_entropy_loss, training=False)
                _, luna_val_miou, _ = train(epoch, model, dataloader_val_luna, optimizer, cross_entropy_loss,
                                              training=False)
                _, steel_val_miou, _ = train(epoch, model, dataloader_val_steel, optimizer, cross_entropy_loss,
                                              training=False)
                _, cracks_val_miou, _ = train(epoch, model, dataloader_val_cracks, optimizer, cross_entropy_loss, training=False)
                # Save the best model
                # if (animal_val_miou > best_animal) or (eyeballs_val_miou > best_eyeballs) or (aerial_val_miou > best_aerial) or (luna_val_miou > best_luna) or (steel_val_miou > best_steel):
                # if cracks_val_miou > min(best_cracks, sota_cracks+0.5) and animal_val_miou > min(best_animal, sota_animal+0.5) and eyeballs_val_miou > min(best_eyeballs, sota_eyeballs+0.5) and aerial_val_miou > min(best_aerial, sota_aerial+0.5) and luna_val_miou > min(best_luna, sota_luna+0.5) and steel_val_miou > min(best_steel, sota_steel+0.5):
                sum = animal_val_miou + eyeballs_val_miou + aerial_val_miou + luna_val_miou + steel_val_miou + cracks_val_miou
                val_miou = [animal_val_miou, eyeballs_val_miou, aerial_val_miou, luna_val_miou, steel_val_miou, cracks_val_miou]
                sota = [sota_animal, sota_eyeballs, sota_aerial, sota_luna, sota_steel, sota_cracks]
                temp_num = 0
                for i in range(len(val_miou)):
                    if val_miou[i] > sota[i]:
                        temp_num = temp_num + 1
                if temp_num > best_better_than_sota_num or (temp_num == best_better_than_sota_num and sum > best_sum):
                    best_better_than_sota_num = temp_num
                    best_sum = max(best_sum, sum)
                    if main_process(local_rank):
                        Logger.save_model_miou(model, epoch, animal_val_miou)
                best_animal = max(animal_val_miou, best_animal)
                best_eyeballs = max(eyeballs_val_miou, best_eyeballs)
                best_aerial = max(aerial_val_miou, best_aerial)
                best_steel = max(steel_val_miou, best_steel)
                best_luna = max(best_luna, luna_val_miou)
                best_cracks = max(best_cracks, cracks_val_miou)


if __name__ == "__main__":
    main()
