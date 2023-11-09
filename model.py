import torch
from torch import nn
from transformer import MultiScaleTransformer
import torch.nn.functional as F
from functools import reduce
from operator import add
from Decoder import fpn
from DCAMA.model.DCAMA import DCAMA
import fvcore.nn.weight_init as weight_init
from feature_visualization import feature_visualization
import os


class Model(nn.Module):
    def __init__(self, backbone, pretrained_path, use_original_imgsize, feat_channels, hidden_dim, num_queries=100, nhead=8, dec_layers=3,
                 conv_dim=256, mask_dim=256, enforce_input_project=False, add_mask=True):
        super(Model, self).__init__()
        self.feat_channels = feat_channels  # [256, 512, 1024]-> [1/8, 1/16. 1/32]
        self.decoder = fpn.FPNDecoder(feature_channels=feat_channels, conv_dim=conv_dim, mask_dim=mask_dim)
        self.feat_extractor = DCAMA(backbone, pretrained_path, mask_dim, use_original_imgsize)
        reverse_feat_channels = self.feat_channels[::-1] # [1024, 512, 256]
        self.predictor = MultiScaleTransformer.MultiScaleMaskedTransformerDecoder(in_channels=reverse_feat_channels[:-1], hidden_dim=hidden_dim,
            num_queries=num_queries, nheads=nhead, dim_feedforward=2048, dec_layers=dec_layers, mask_dim=mask_dim, enforce_input_project=enforce_input_project, add_mask=add_mask)
        outch1, outch2, outch3 = 16, 64, 128
        # mixer blocks
        self.mixer1 = nn.Sequential(
            nn.Conv2d(mask_dim+num_queries, outch3, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

    def forward(self, query_img, support_img, support_mask, nshot):
        # len(s_features) == len(q_features) = 3, len(q_features) = 4
        # from high resolution to low resolution
        s_features, q_features, q_masks, mask_features, support_mask = self.feat_extractor(query_img, support_img, support_mask, nshot)
        # feature_visualization(q_features[0], 384, 50, os.path.join('./features', 'backbone'))
        # feature_visualization(mask_features, 384, 50, os.path.join('./features', 'dcama'))
        # q_features from low resolution to high resolution
        _, q_features = self.decoder(q_features)  # (bsz, mask_dim, 1/4*H, 1/4*W)

        if nshot == 1:
            mask = support_mask.unsqueeze(1)
            s_features = s_features[::-1]
            q_masks = q_masks[::-1]
            s_masks = []
            for i in range(len(s_features)):
                _, _, h, w = s_features[i].shape
                s_masks.append(F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=True))
                # assert s_features[i].shape == q_features[i].shape
                assert q_masks[i].shape[-2] == q_features[i].shape[-2]
                assert s_features[i].shape[-2] == s_masks[i].shape[-2]
                assert q_features[i].shape[-2] == s_features[i].shape[-2]
        else:
            q_masks = q_masks[::-1]
            s_masks = []
            for s_k in range(nshot):
                mask_shot = support_mask[:, s_k].unsqueeze(1)
                s_features[s_k] = s_features[s_k][::-1]
                s_level_mask_shot = [F.interpolate(mask_shot, size=s_features[s_k][l_i].shape[-2], mode='bilinear', align_corners=True) for l_i in range(len(s_features[s_k]))]
                s_masks.append(s_level_mask_shot)

        output, outputs_mask =self.predictor(s_features, q_features, s_masks, q_masks, mask_features, nshot)
        # output_mask = (bsz, queries, 1/4*H, 1/4*W)
        # output = (bsz, queries, hidden_dim)

        out = torch.cat((mask_features, outputs_mask), dim=1) # (bsz, queries+mask_dim, 1/4*H, 1/4*W)
       
        out = self.mixer1(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True) # 1/2*H, W
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)

        return logit_mask


if __name__ == "__main__":
    model = Model('swin-b', pretrained_path='swin_base_patch4_window12_384_22kto1k.pth', use_original_imgsize=False,
                  feat_channels=[128, 256, 512, 1024], hidden_dim=256, conv_dim=256, mask_dim=256, num_queries=100,
                  nhead=8, add_mask=False)
    query_img = torch.randn(1, 3, 384, 384)
    support_img = torch.randn(1, 3, 384, 384)
    support_mask = torch.randn(1, 384, 384)
    output = model(query_img, support_img, support_mask)
    print(output.shape)








