r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torch.utils.data.distributed import DistributedSampler as Sampler
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.custom import DatasetCustom
from data.cityscapes import DatasetCityscapes
from data import getClasses


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'cityscapes': DatasetCityscapes
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        # cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, datapath, test_num, distributed, benchmark, bsz, nworker, fold, split, shot=1, training=True):
        nworker = nworker if split == 'trn' else 0
        if training:
            dataset = cls.datasets[benchmark](datapath, fold=fold,
                                              transform=cls.transform,
                                              split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        else:
            # benchmark相当于datasetname
            cls_num, cls_names = getClasses.get_classes(datapath, shot, benchmark)

            dataset = DatasetCustom(
                base_dir=datapath,
                n_shots=shot,
                transform=cls.transform,
                max_iters=test_num,
                dataset_name=benchmark,
                class_names=cls_names,
                use_original_imgsize=cls.use_original_imgsize
            )
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        if distributed:
            train_sampler = Sampler(dataset) if split == 'trn' else None
        else:
            train_sampler = RandomSampler(dataset) if split == 'trn' else None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, sampler=train_sampler, num_workers=nworker,
                                pin_memory=False)

        return dataloader
