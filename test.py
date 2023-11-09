import torch.nn as nn
import torch

from model import Model
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
import os
import neptune
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10000"
API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYWYzMTAyMy1jMzkyLTQzZGYtOThiMC0xZWIxMmZhODU3OTIifQ=="


def test(model, dataloader, nshot):
    r""" Test """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)

    # AverageMeter内部有参数要改
    average_meter = AverageMeter(dataloader.dataset)
    epoch = args.test_epoch
    for i in range(args.test_epoch):
        for idx, batch in enumerate(dataloader):

            # 1. forward pass
            batch = utils.to_cuda(batch)
            if args.nshot == 1:
                logit_mask = model.module(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1), 1)
                pred_mask = logit_mask.argmax(dim=1)    
            else:
                if (not args.vote) and (not args.average):
                    logit_masks = []
                    for k in range(args.nshot):
                        temp_mask = model.module(batch['query_img'], batch['support_imgs'][:, k], batch['support_masks'][:, k], 1)
                        logit_masks.append(temp_mask)
                    logit_mask = torch.stack(logit_masks, dim=1).mean(dim=1)
                    pred_mask = logit_mask.argmax(dim=1)
                elif args.vote:
                    logit_mask_agg = 0
                    for s_idx in range(args.nshot):
                        logit_mask = model.module(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx], 1)
                        logit_mask_agg += logit_mask.argmax(dim=1).clone()
                    
                    bsz = logit_mask_agg.size(0)
                    max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
                    max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
                    max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
                    pred_mask = logit_mask_agg.float() / max_vote
                    pred_mask[pred_mask < 0.5] = 0
                    pred_mask[pred_mask >= 0.5] = 1
                elif args.average:
                    logit_mask = model.module(batch['query_img'],batch['support_imgs'], batch['support_masks'], args.nshot)
                    pred_mask = logit_mask.argmax(dim=1)

            assert pred_mask.size() == batch['query_mask'].size()

            # 2. Compute loss & update model parameters
            # loss = model.module.compute_objective(logit_mask, batch['query_mask'])
            bsz = logit_mask.size(0)
            logit_mask = logit_mask.view(bsz, 2, -1)
            gt_mask = batch['query_mask'].view(bsz, -1).long()

            # 2. Evaluate prediction
            area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)

            average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), epoch=i, write_batch_idx=1)

            # Visualize predictions
            if Visualizer.visualize:
                Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                      batch['query_img'], batch['query_mask'],
                                                      pred_mask, batch['class_id'], idx,
                                                      iou_b=area_inter[1].float() / area_union[1].float())
            torch.cuda.empty_cache()

    # Write evaluation results
    average_meter.write_result('Test', epoch)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    if args.neptune:
        run = neptune.init_run(project="wujr/Model", api_token=API_TOKEN, source_files=["**/*.py"])
        run["parameters"] = args
        run["sys/tags"].add("test")
        run["sys/tags"].add(args.test_dataset)
        run["sys/tags"].add(str(args.nshot)+"shot")

    Logger.initialize(args, training=False)

    # Model initialization
    if args.backbone == 'swin-b':
        _feat_channels = [128, 256, 512, 1024]
    if args.backbone == 'swin-l':
        _feat_channels = [192, 384, 768, 1536]
    if args.backbone == 'resnet101' or args.backbone == 'resnet50':
        _feat_channels = [256, 512, 1024, 2048]
    model = Model(backbone=args.backbone, pretrained_path=args.feature_extractor_path, use_original_imgsize=False, feat_channels=_feat_channels, hidden_dim=args.hidden_dim,
            num_queries=args.num_queries, nhead=args.nhead, dec_layers=args.dec_layers, conv_dim=args.conv_dim, mask_dim=args.mask_dim, enforce_input_project=args.enforce_input_project,
            add_mask=args.add_mask)

    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    params = model.state_dict()
    state_dict = torch.load(args.load)
    # state_dict directly from saved model
    # params from the above constructed model
    # complete test
    
    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)
    """
    new_state_dict = model.state_dict()
    flag = False 
    for key1 in list(params.keys()):
        
        print("key1", key1)
        flag = False
        if 'cross_attention_layers.0' in key1:
            for key2 in list(state_dict.keys()):
                print("key2", key2)
                if key2 == key1.replace('0', '2'):
                    new_state_dict[key1] = state_dict.pop(key2)
                    flag = True
                    break
        elif 'ffn_layers.0' in key1:
            for key2 in list(state_dict.keys()):
                print("key2", key2)
                if key2 == key1.replace('0', '2'):
                    new_state_dict[key1] = state_dict.pop(key2)
                    flag = True
                    break
        else:
            for key2 in list(state_dict.keys()):
                print("key2", key2)
                if key1 == key2:
                    new_state_dict[key1] == state_dict.pop(key2)
                    flag = True
                    break
        if flag == False:
            raise ValueError("Match Failure!")
        
        for key2 in list(params.keys()):
            if key1 == key2:
                new_state_dict[key1] = state_dict.pop(key2)
    """


    model.load_state_dict(state_dict)
    model.module.eval()

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    # Dataset initialization
    FSSDataset.initialize(img_size=384, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.test_datapath, args.test_num, args.distributed, args.test_dataset, args.bsz, args.nworker, args.fold, 'val', args.nshot, training=False)
    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    if args.neptune:
        run["result"] = test_miou
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
