import os, io
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np


class CovidCTDataset(Dataset):
    def __init__(self,mode, root_dir, info_csv,transform=None):
        """
        Args:
            info_csv (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - img1.png
            - img2.png
            - ......

        """
        self.root_dir = root_dir
        self.classes = ['covid', 'normal']
        self.num_cls = len(self.classes)
        self.img_list = []
        self.full_volume = None
        self.affine = None
        self.label_df = pd.read_csv(info_csv)
        for line in self.label_df:
            cls_list = [os.path.join(root_dir, line["filename"]), self.classes.index(line["class"])]
            self.img_list += cls_list

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        val_transformer = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        if(mode == 'train'):

            self.transform = train_transformer

        else:
            self.transform = val_transformer
        print('samples = ', len(self.img_list))


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]

        image = Image.open(img_path).convert('RGB')

        with open(img_path, 'rb') as f:
            tif = Image.open(io.BytesIO(f.read()))
        image = np.array(tif)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(int(self.img_list[idx][1]), dtype=torch.long)
