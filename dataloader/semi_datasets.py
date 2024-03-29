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

class SemiDataset(data.Dataset):
    def __init__(self, cfg, phase="train", last_dataset=None):
        super().__init__()

        self.label_percentage = cfg.semi.label_percentage
        self.is_wd = cfg.semi.with_distractors
        if self.is_wd:
            self.distractor_class = cfg.semi.distractor_class
        else:
            self.distractor_class = 0
        self.transform = self.prepare_transform(cfg, phase)

        if phase == "train":
            self.unlabeled_per_task = cfg.semi.unlabeled_per_task_train
        else:
            self.unlabeled_per_task = cfg.semi.unlabeled_per_task_test
        
        if last_dataset is not None:
            self.class_data_dict = last_dataset.class_data_dict # semi-fsl requires only split once
        else:
            self.class_data_dict = self.prepare_class_split(cfg, phase)
        self.data_list = self.prepare_data_list(cfg, phase)
    
    def prepare_transform(self, cfg, phase):
        norm = transforms.Normalize(
            np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
        )
        if cfg.model.encoder == "FourLayer_64F":
            if phase == "train":
                t = [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]
        else:
            if phase == "train":
                t = [
                    transforms.Resize([92, 92]),
                    transforms.RandomCrop(84),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    norm
                ]
        return transforms.Compose(t)

    def prepare_class_split(self, cfg, phase):
        folder = osp.join(cfg.data.image_dir, phase)
        
        class_folders = [osp.join(folder, label) \
            for label in os.listdir(folder) \
            if osp.isdir(osp.join(folder, label)) \
        ]
        random.shuffle(class_folders)
        
        class_data_dict = {}
        for f in class_folders:
            f_all_images = [osp.join(f, img) for img in os.listdir(f) if ".png" in img or ".jpg" in img]
            random.shuffle(f_all_images)
            labeled_images = f_all_images[:int(len(f_all_images) * self.label_percentage)]
            unlabeled_images = f_all_images[int(len(f_all_images) * self.label_percentage): ]
            class_data_dict[osp.basename(f)] = {"labeled": labeled_images, "unlabeled": unlabeled_images}
        return class_data_dict

    def prepare_data_list(self, cfg, phase):

        class_data_dict = self.class_data_dict
        class_list = class_data_dict.keys()

        data_list = []
        query_per_class_per_episode = cfg.train.query_per_class_per_episode if phase == "train" else cfg.test.query_per_class_per_episode
        if phase == "train":
            episode_per_epoch = cfg.train.episode_per_epoch
        elif phase == "val":
            episode_per_epoch = cfg.val.episode
        else:
            episode_per_epoch = cfg.test.episode
        for e in range(episode_per_epoch):
            episode_labeled = []
            episode_unlabeled = []
            classes = random.sample(class_list, cfg.n_way + self.distractor_class)
            for t, c in enumerate(classes[:cfg.n_way]):
                labeled_set = class_data_dict[c]["labeled"]
                labeled_select = random.sample(labeled_set, cfg.k_shot + query_per_class_per_episode)
                random.shuffle(labeled_select)
                support_x = labeled_select[:cfg.k_shot]
                query_x = labeled_select[cfg.k_shot:]
                episode_labeled.append({ "support_x": support_x, "query_x": query_x, "target": t })

                unlabeled_select = random.sample(class_data_dict[c]["unlabeled"], self.unlabeled_per_task)
                episode_unlabeled += unlabeled_select
            for c in classes[cfg.n_way:]:
                disctractor_select = random.sample(class_data_dict[c]["unlabeled"], self.unlabeled_per_task)
                episode_unlabeled += disctractor_select
            random.shuffle(episode_unlabeled)
            data_list.append((episode_labeled, episode_unlabeled))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        episode_labeled, episode_unlabeled = self.data_list[index]
        support_x, support_y, query_x, query_y, unlabeled_x = [], [], [], [], []
        for e in episode_labeled:
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

        for u in episode_unlabeled:
            im = self.transform(Image.open(u).convert("RGB"))
            unlabeled_x.append(im.unsqueeze(0))
        unlabeled_x = torch.cat(unlabeled_x, 0)

        support_x = torch.cat(support_x, 0)
        query_x = torch.cat(query_x, 0)
        support_y = torch.LongTensor(support_y)
        query_y = torch.LongTensor(query_y)

        randperm = torch.randperm(len(query_y))
        query_x = query_x[randperm]
        query_y = query_y[randperm]
        return support_x, support_y, query_x, query_y, unlabeled_x

