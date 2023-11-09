import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms


# 只用来训练，不需要验证集
class DatasetADE20k(Dataset):
    def __init__(self, datapath, transform, split, shot, use_original_imgsize):
        self.split = split
        self.nclass = 150
        self.benchmark = 'ade20k'
        self.shot = shot
        self.base_path = os.path.join(datapath, 'ADEChallengeData2016')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:],
                                       mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                                mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_class_ids(self):
        class_ids = [x for x in range(self.nclass)]
        return class_ids

    def build_img_metadata_classwise(self):
        with open('./splits/ade20k/%s/ade20k.pkl' % self.split, 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    # ADE_train_00000024_1
    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, str):
        mask_name = str + '.png'
        mask_path = os.path.join(self.base_path, 'annotations', 'training', mask_name)
        mask = torch.tensor(np.array(Image.open(mask_path)))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_str = np.random.choice(self.img_metadata_classwise[class_sample + 1], 1, replace=False)[0]
        query_str = query_str[:query_str.find('_', 10)]
        query_name = query_str + '.jpg'
        query_img = Image.open(os.path.join(self.base_path, 'images', 'training', query_name)).convert('RGB')
        query_mask = self.read_mask(query_str)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        support_strs = []
        while True:  # keep sampling support set if query == support
            support_str = np.random.choice(self.img_metadata_classwise[class_sample + 1], 1, replace=False)[0]
            support_str = support_str[:support_str.find('_', 10)]
            support_name = support_str + '.jpg'
            if query_name != support_name:
                support_names.append(support_name)
                support_strs.append(support_str)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []

        for (support_str, support_name) in zip(support_strs, support_names):
            support_imgs.append(Image.open(os.path.join(self.base_path, 'images', 'training', support_name)).convert('RGB'))
            support_mask = self.read_mask(support_str)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize

if __name__ == "__main__":
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(size=(384, 384)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(img_mean, img_std)])
    dataset = DatasetADE20k(datapath=r'D:\ADEChallengeData2016', transform=transform, split='trn', shot=1, use_original_imgsize=False)
    for i in range(100):
        print(dataset[i]['class_id'])
        print(dataset[i]['query_name'])
        print(dataset[i]['support_names'])