import torch
from torch import nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init


class FPNDecoder(nn.Module):
    def __init__(self, feature_channels, conv_dim, mask_dim):
        super().__init__()

        norm = ""

        self.in_features = [0, 1, 2, 3]

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=False, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
                # Place convs into top-down order (from low to high resolution)
                # to make the top-down computation in forward clearer.
            self.lateral_convs = lateral_convs[::-1]
            self.output_convs = output_convs[::-1]

            self.mask_dim = mask_dim
            self.mask_features = Conv2d(
                conv_dim,
                mask_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            weight_init.c2_xavier_fill(self.mask_features)

            self.maskformer_num_feature_levels = 3  # always use 3 scales


    def forward(self, features):
            multi_scale_features = []
            num_cur_levels = 0
            # Reverse feature maps into top-down order (from low to high resolution)
            for idx, f in enumerate(self.in_features[::-1]): # [3, 2, 1, 0]
                x = features[f]
                lateral_conv = self.lateral_convs[idx]
                output_conv = self.output_convs[idx]
                if lateral_conv is None:
                    y = output_conv(x)
                else:
                    cur_fpn = lateral_conv(x)
                    # Following FPN implementation, we use nearest upsampling here
                    y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                    y = output_conv(y)
                if num_cur_levels < self.maskformer_num_feature_levels:
                    multi_scale_features.append(y)
                    num_cur_levels += 1
            return self.mask_features(y), multi_scale_features


if __name__ == "__main__":
    feature0 = torch.randn(1, 128, 96, 96)
    feature1 = torch.randn(1, 256, 48, 48)
    feature2 = torch.randn(1, 512, 24, 24)
    feature3 = torch.randn(1, 1024, 12, 12)
    fpn_model = FPNDecoder(feature_channels=[128, 256, 512, 1024], conv_dim=256, mask_dim=256)
    mask_features, multi_scale_features = fpn_model([feature0, feature1, feature2, feature3])
    for i in range(len(multi_scale_features)):
        print(multi_scale_features[i].shape)
