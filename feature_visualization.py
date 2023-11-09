import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import os
import math
import torch.nn.functional as F

# 分别插在backbone， dcama mask和最终的mask后面

def feature_visualization(features, img_size, feature_num, save_dir):
    """
    features: The feature map which you need to visualization [bsz, ch, h, w]
    model_type: The type of feature map
    model_id: The id of feature map
    feature_num: The amount of visualization you need
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features = F.interpolate(features, size=(img_size, img_size), mode='bilinear')

    # print(features.shape)
    # block by channel dimension
    blocks = torch.chunk(features, features.shape[1], dim=1)

    # # size of feature
    # size = features.shape[2], features.shape[3]

    for i in range(feature_num):
        torch.squeeze(blocks[i]) #[1, h, w]
        feature = transforms.ToPILImage()(blocks[i].squeeze())

        plt.imshow(feature)
        fig = plt.gcf()
        fig.savefig(os.path.join(save_dir, 'feature_{}.png'.format(i)), dpi=300)
        # gray feature
        # plt.imshow(feature, cmap='gray')

