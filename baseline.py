import torch
from torch import nn
from transformer import MultiScaleTransformer
import torch.nn.functional as F
from functools import reduce
from operator import add
from Decoder import Decoder
from DCAMA.model.DCAMA import DCAMA
import fvcore.nn.weight_init as weight_init


class Model(nn.Module):
    def __init__(self, backbone, pretrained_path, use_original_imgsize, mask_dim=256):
        super(Model, self).__init__()
        self.feat_extractor = DCAMA(backbone, pretrained_path, mask_dim, use_original_imgsize)
        outch1, outch2, outch3 = 16, 64, 128
        # mixer blocks
        self.mixer1 = nn.Sequential(
            nn.Conv2d(192, outch3, (3, 3), padding=(1, 1), bias=True),
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

    def forward(self, query_img, support_img, support_mask, nshot=1):
        # len(s_features) == len(q_features) = 3, len(q_features) = 4
        # from high resolution to low resolution
        _, q_features, _,_ ,_= self.feat_extractor(query_img, support_img, support_mask, 1)
        # for i in range(len(q_features)):
        #     print(q_features[i].shape)
        out = self.mixer1(q_features[0])
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








