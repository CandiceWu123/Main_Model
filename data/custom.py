# PairedDataset可以直接使用

from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F


class DatasetCustom(Dataset):
    def __init__(self, base_dir, n_shots, transform, max_iters, dataset_name, class_names, use_original_imgsize):
        super().__init__()
        # 包含所有数据集的根目录
        self.base_dir = base_dir
        self.ids = []
        self.shots = n_shots
        self.queries = 1
        # self.categories = [i for i, _ in enumerate(self.cls_names)]
        self.max_iters = max_iters
        self.cls_names = class_names
        # 输入数据集的名称
        self.dataset_name = dataset_name
        self.benchmark = dataset_name
        self.transform = transform
        self.class_ids = [id for id in range(len(self.cls_names))]
        self.use_original_imgsize = use_original_imgsize
        # print(self.class_ids)

        temp_ids = []
        cls_ids = []

        # base_dir = C:\Users\86189\Desktop\数据集3.0
        # 直接为support和query上一级的目录
        if dataset_name == 'Animal' or dataset_name == 'Magnetic_tile_surface' or dataset_name == 'Artificial_Luna_Landscape' or dataset_name == 'Aerial':
            # dataset_path = os.path.join(self.base_dir, dataset_name)
            # for cls_name in os.listdir(dataset_path):
            #    if len(os.listdir(os.path.join(dataset_path, cls_name,'support','images'))) >= self.shots:
            #        self.cls_names.append(cls_name)
            self.path_dir = os.path.join(self.base_dir, dataset_name)
        else:
            # self.cls_names = [self.dataset_name]
            self.path_dir = self.base_dir

        for name in self.cls_names:
            for id in os.listdir(os.path.join(self.path_dir, name,'support','images')):
                temp_ids.append(id)
            cls_ids.append(temp_ids)
            temp_ids = []
            for id in os.listdir(os.path.join(self.path_dir, name,'query','images')):
                temp_ids.append(id)
            cls_ids.append(temp_ids)
            self.ids.append(cls_ids)
            cls_ids = []
            temp_ids = []

    def __len__(self):
        # 返回有多少张query图，其他类比的肯定少于1000，animal不一定，需要计算
        return self.max_iters

    def __getitem__(self, idx):
        support_names, query_name, class_sample = self.sample_episode(idx)
        # print("support_names: ")
        # print(support_names)
        # print("query_names: ")
        # print(query_name)

        # 先处理query image和mask
        query_mask_name = query_name.split('.')[0] + ".png"
        query_img = Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'query', 'images', query_name))
        org_qry_imsize = query_img.size
        query_img = self.transform(self.process_image(query_img))
        # print(np.unique(query_img))
        query_mask = Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'query', 'masks', query_mask_name))
        query_mask = self.process_mask(query_mask, class_sample)
        if not self.use_original_imgsize:
            # (384, 384)
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_ignore_idx = self.extract_ignore_idx(query_mask)

        # 处理support images和mask
        support_mask_names = [name.split('.')[0] + ".png" for name in support_names]
        # support_imgs = [cv2.imread(os.path.join(self.path_dir, self.cls_names[class_id], 'support', 'images', name),
        #                          cv2.IMREAD_COLOR) for name in support_names]
        # support_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in support_imgs]
        support_imgs = [Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'support', 'images', name)) for name in support_names]
        support_imgs = torch.stack([self.transform(self.process_image(support_img)) for support_img in support_imgs])

        support_cmasks = [Image.open(os.path.join(self.path_dir, self.cls_names[class_sample], 'support', 'masks', name)) for name in
                        support_mask_names]
        support_cmasks = [self.process_mask(mask, class_sample) for mask in support_cmasks]
        support_ignore_idxs = []
        support_masks = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                   mode='nearest').squeeze()
            support_masks.append(scmask)
            support_ignore_idx = self.extract_ignore_idx(scmask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': [query_name],
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 # support_names有问题
                 'support_ignore_idxs': support_ignore_idxs,
                 'class_id': torch.tensor(class_sample)
                 }
        # print([query_name])
        # print(support_names)
        return batch

    # 所有mask都是只有黑白两种颜色的灰度图像，白色表示分割目标，值为255，黑色表示背景，值为0
    def extract_ignore_idx(self, mask):
        boundary = (mask / 255).floor()
        return boundary

    def process_mask(self, label, class_id):
        if label.mode != 'L':
            label = label.convert('L')
        label = torch.tensor(np.array(label))
        label = torch.where(label == 255, torch.ones_like(label), torch.zeros_like(label))
        return label

    def process_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    def sample_episode(self, idx):
        # 本次采样的类别
        class_id = idx % len(self.cls_names)
        support_names = np.random.choice(self.ids[class_id][0], self.shots, replace=False)
        query_name = np.random.choice(self.ids[class_id][1], 1, replace=False)
        return support_names, query_name[0], class_id












