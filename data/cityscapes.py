import os
import pickle
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms

class DatasetCityscapes(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = split
        self.benchmark = 'cityscapes'
        self.shot = shot
        self.nclass = 19
        self.base_path = datapath
        self.use_original_imgsize = use_original_imgsize
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = [x for x in range(self.nclass)]
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize, class_sample, query_name, support_names = self.load_frame()
        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                   mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)

        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask == class_id] = 100
        mask[mask != 100] = 0
        mask[mask == 100] = 1

        return mask, boundary

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_mask_path = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_mask = self.read_mask(query_mask_path)
        query_name = query_mask_path[6:][:-24]
        query_img_path = "leftImg8bit" + query_name + "leftImg8bit.png"
        query_img = Image.open(os.path.join(self.base_path, query_img_path)).convert('RGB')

        support_mask_paths = []
        while True:
            support_mask_path = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if support_mask_path != query_mask_path:
                support_mask_paths.append(support_mask_path)
                if len(support_mask_paths) == self.shot: break

        support_masks = [self.read_mask(mask) for mask in support_mask_paths]
        support_names = [mask[6:][:-24] for mask in support_mask_paths]
        support_img_paths = ["leftImg8bit" + name + "leftImg8bit.png" for name in support_names]
        support_imgs = [Image.open(os.path.join(self.base_path, img_path)).convert('RGB') for img_path in support_img_paths]

        org_qry_imsize = query_img.size

        return  query_img, query_mask, support_imgs, support_masks, org_qry_imsize, class_sample, query_name, support_names


    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, name)
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        return mask

    def build_img_metadata_classwise(self):
        with open('./data/splits/cityscapes/cityscapes.pkl', 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))


if __name__ == "__main__":
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(size=(384, 384)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(img_mean, img_std)])
    dataset = DatasetCityscapes(datapath=r'D:\train_dataset', fold=0, transform=transform, split='trn', shot=1, use_original_imgsize=False)

    """
    class_sample = 7
    mask = dataset.read_mask("gtFine/train/cologne/cologne_000116_000019_gtFine_labelTrainIds.png")
    print(np.unique(mask))
    query_mask, query_ignore_idx = dataset.extract_ignore_idx(mask.float(), class_sample)
    print(np.unique(query_mask))
    
    """
    for i in range(500):
        print("idx", i)
        batch = dataset[i]
        print("class_sample", batch['class_id'])
        class_id = batch['class_id'].item()
        if class_id == 0:
            print(np.unique(batch['query_mask']))
            for j in range(len(batch['support_masks'])):
                print(np.unique(batch['support_masks'][j]))



