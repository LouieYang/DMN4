import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
import csv

class BalancedDataset(data.Dataset):
    def __init__(self, cfg, phase="train"):
        super().__init__()

        self.data_list = self.prepare_data_list(cfg, phase)
        size = 84
        if phase == "train" and cfg.train.data_aug == 1:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def prepare_data_list(self, cfg, phase):
        folder = osp.join(cfg.data.root, phase)
        
        class_folders = [osp.join(folder, label) \
            for label in os.listdir(folder) \
            if osp.isdir(osp.join(folder, label)) \
        ]
        random.shuffle(class_folders)
        
        class_img_dict = {
            osp.basename(f): [osp.join(f, img) for img in os.listdir(f) if ".png" in img] \
            for f in class_folders
        }
        class_list = class_img_dict.keys()
        query_per_class_per_episode = cfg.train.query_per_class_per_episode if phase == "train" else cfg.test.query_per_class_per_episode
        episode_per_epoch = cfg.train.episode_per_epoch if phase == "train" else cfg.test.episode
        groups = len(class_list) // cfg.n_way
        balanced_splits = episode_per_epoch // groups

        data_list = []
        for s in range(balanced_splits):
            class_list_copy = list(class_img_dict)
            random.shuffle(class_list_copy)
            for i in range(groups):
                episode = []
                classes = class_list_copy[i * cfg.n_way: (i + 1) * cfg.n_way]
                for t, c in enumerate(classes):
                    imgs_set = class_img_dict[c]
                    imgs_select = random.sample(imgs_set, cfg.k_shot + query_per_class_per_episode)
                    random.shuffle(imgs_select)
                    support_x = imgs_select[:cfg.k_shot]
                    query_x = imgs_select[cfg.k_shot:]

                    episode.append({
                        "support_x": support_x,
                        "query_x": query_x,
                        "target": t
                    })
                data_list.append(episode)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        episode = self.data_list[index]
        support_x, support_y, query_x, query_y = [], [], [], []
        for e in episode:
            query_ = e["query_x"]
            for q in query_:
                im = self.transform(Image.open(q).convert("RGB"))
                query_x.append(im.unsqueeze(0))
            support_ = e["support_x"]
            for s in support_:
                im = self.transform(Image.open(s).convert("RGB"))
                support_x.append(im.unsqueeze(0))
            target = e["target"]
            support_y.extend(np.tile(target, len(support_)))
            query_y.extend(np.tile(target, len(query_)))

        support_x = torch.cat(support_x, 0)
        query_x = torch.cat(query_x, 0)
        support_y = torch.LongTensor(support_y)
        query_y = torch.LongTensor(query_y)

        randperm = torch.randperm(len(query_y))
        query_x = query_x[randperm]
        query_y = query_y[randperm]
        return support_x, support_y, query_x, query_y
