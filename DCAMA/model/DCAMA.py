r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from .base.swin_transformer import SwinTransformer
from .base.transformer import MultiHeadedAttention, PositionalEncoding


class DCAMA(nn.Module):

    def __init__(self, backbone, pretrained_path, mask_dim, use_original_imgsize):
        super(DCAMA, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.m_nlayers = [4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.m_nlayers = [4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin-b':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
            self.m_nlayers = [2, 18, 2]
        elif backbone == 'swin-l':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, num_classes=21841, window_size=12, embed_dim=192, depths=[2, 2, 18, 2],
                    num_heads=[6, 12, 24, 48])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [192, 384, 768, 1536]
            self.nlayers = [2, 2, 18, 2]
            self.m_nlayers = [2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)

        self.m_lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.m_nlayers)])
        self.m_stack_ids = torch.tensor(self.m_lids).bincount()[-3:].cumsum(dim=0)

        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, mask_dim=mask_dim)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask, nshot):
        if nshot == 1:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                support_feats = self.extract_feats(support_img)
            query_masks, mask_features = self.model(query_feats, support_feats, support_mask.clone(), nshot=nshot)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                n_support_feats = []
                # n_support_masks = []
                for k in range(nshot):
                    support_feat = self.extract_feats(support_img[:, k]) # 一维列表，共包含x个特征
                    n_support_feats.append(support_feat) # [(batchsize, ch, h, w), (batchsize, ch, h, w)..., (batchsize, ch, h, w)] 二维列表
                    # n_support_masks.append(support_mask[:, k])
                query_masks, mask_features = self.model(query_feats, n_support_feats, support_mask.clone(), nshot=nshot)
                """
                # mask2former
                temp = torch.stack(n_support_masks, dim=1) # [batchsize, nshot, h, w]
                support_mask = temp.sum(dim=1) / nshot # return support_mask
                support_feats = []
                for i in range(len(n_support_feats[0])):
                    feat = torch.stack([n_support_feats[j][i] for j in range(nshot)], dim=1)
                    feat = torch.sum(feat, dim=1) / nshot
                    # print("feat", feat.shape)
                    support_feats.append(feat)
                """
        # s_features包括1/8, 1/16和1/32
        if nshot == 1:
            s_features = [torch.mean(torch.stack(support_feats[start: end], dim=1), dim=1) for start, end in zip(self.stack_ids[:-1], self.stack_ids[1:])]
        else:
            s_features = []
            for k in range(nshot):
                support_feats_shot = n_support_feats[k]
                s_level_feats_shot = [torch.mean(torch.stack(support_feats_shot[start: end], dim=1), dim=1) for start, end in zip(self.stack_ids[:-1], self.stack_ids[1:])]
                s_features.append(s_level_feats_shot) # 二维列表 [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

        # q_features包括1/4， 1/8，1/16和1/32，1/4没有使用
        q_features = [torch.mean(torch.stack(query_feats[start: end], dim=1), dim=1) for start, end in zip(torch.cat((torch.tensor([0]), self.stack_ids[:-1])), self.stack_ids[:])]
        q_masks = [torch.mean(torch.stack(query_masks[start: end], dim=1), dim=1) for start, end in zip(torch.cat((torch.tensor([0]), self.m_stack_ids[:-1])), self.m_stack_ids[:])]

        # if nshot > 1, support_mask's shape=(bsz, nshot, h, w)
        return s_features, q_features, q_masks, mask_features, support_mask

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin-b' or self.backbone == 'swin-l':
            
            _ = self.feature_extractor.forward_features(img)
            i = 1
            
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                ## print(hw)
                # print(c)
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                # print(feat.shape)
                feats.append(feat)
                i = i + 1
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            # [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2]
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0 经过第一层(conv1的提取)
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)
            # output feat: (1, 64, 96, 96)
            # bottleneck_ids  =     [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2]
            # Layer 1-4 self.lids = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                # __getattr__ 是一个对象方法，找不到对象属性时调用这个方法
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                # 上面这一步的输出的维度为256
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids, mask_dim):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids

        # DCAMA blocks:
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        self.mixer = nn.Sequential(nn.Conv2d(outch3+2*in_channels[1]+2*in_channels[0], mask_dim, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(mask_dim, mask_dim, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())


    def forward(self, query_feats, support_feats, support_mask, nshot):
        coarse_masks = []
        # 分别计算每一个尺度的query feature
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            # (1, 2304, 512)
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous() # bsz, ha, wa, ch
            if nshot == 1:
                support_feat = support_feats[idx]
                mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                     align_corners=True).view(support_feat.size()[0], -1)
                support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                support_feat = support_feat.permute(1, 0, 2, 3, 4)
                # (5, 2304, 512)
                support_feat = support_feat.contiguous().view(-1, ch, ha * wa).permute(0, 2, 1)
                mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                    for k in support_mask])
                # (1, 11520)
                mask = mask.view(bsz, -1)

            # DCAMA blocks forward
            if idx < self.stack_ids[1]:
                # coarse_mask (1, 2304, 1) 取8个head的平均值
                coarse_mask = self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask)
            elif idx < self.stack_ids[2]:
                coarse_mask = self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask)

            else:
                coarse_mask = self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask)

            # Multi-head attention layers
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))

            # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3] - 1 - self.stack_ids[0]].size()
        # (1, 3, 12, 12)
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2] - self.stack_ids[0]:self.stack_ids[3] - self.stack_ids[0]]).transpose(0,
        1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2] - 1 - self.stack_ids[0]].size()
        # (1, 6, 24, 24)
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1] - self.stack_ids[0]:self.stack_ids[2] - self.stack_ids[0]]).transpose(0, 1).contiguous().view(
        bsz, -1, ha, wa)

        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1] - 1 - self.stack_ids[0]].size()
        # (1, 4, 48, 48)
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1] - self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)

        # 经过conv block x 3
        # (1, 128, 12, 12)
        coarse_masks1 = self.conv1(coarse_masks1)

        # (1, 128, 24, 24)
        coarse_masks2 = self.conv2(coarse_masks2)

        # (1, 128, 48, 48)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        # 增大尺寸与下一个mask相加
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        mix = coarse_masks1 + coarse_masks2
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3
        # (1, 1152, 48, 48)
        mix = self.conv5(mix) # (1, 128, 48, 48)

        # skip connect 1/8 and 1/4 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(
                dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1) # (1, 640, 48, 48)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True) # (1, 640, 96, 96)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(
                dim=0).values

        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1) # (1, 896, 96, 96)
        output = self.mixer(mix) # (1, mask_dim, 96, 96)
        # 找到mix的参数

        return coarse_masks, output

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)


if __name__ == "__main__":
    model = DCAMA(backbone="swin-b", pretrained_path="../../swin_base_patch4_window12_384_22kto1k.pth", use_original_imgsize=True)
    query_img = torch.randn(1, 3, 384, 384)
    support_img = torch.randn(1, 3, 384, 384)
    support_mask = torch.randn(1, 384, 384)
    s_feature, q_feature, q_masks = model(query_img, support_img, support_mask)


