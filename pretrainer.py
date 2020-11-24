import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import random
import tqdm
from modules.pretrain_model import make_pretrain_model
from modules.fsl_query import make_fsl
from dataloader import make_predataloader
from utils import mean_confidence_interval, AverageMeter, set_seed

class Pretrainer(object):
    def __init__(self, cfg, checkpoint_dir):

        self.prefix = osp.basename(checkpoint_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir

        self.epochs = cfg.pre.epochs

        self.model = make_pretrain_model(cfg).to(self.device)

        self.lr = cfg.pre.lr
        self.lr_decay = cfg.pre.lr_decay
        self.lr_decay_milestones = cfg.pre.lr_decay_milestones

        self.optim = SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=cfg.train.sgd_mom, 
            weight_decay=cfg.train.sgd_weight_decay,
            nesterov=True
        )
        self.lr_scheduler = MultiStepLR(self.optim, milestones=self.lr_decay_milestones, gamma=self.lr_decay)
        self.fsl = make_fsl(cfg).to(self.device)

        self.cfg = cfg
        self.cfg.val.episode = cfg.pre.val_episode

    def save_model(self, postfix=None):
        self.fsl.encoder.load_state_dict(self.model.encoder.state_dict())
        self.fsl.eval()
        filename = "e0_pre.pth" if postfix is None else "e0_pre_{}.pth".format(postfix)
        filename = osp.join(self.checkpoint_dir, filename)
        state = {
            'fsl': self.fsl.state_dict()
        }
        torch.save(state, filename)

    def train(self, dataloader, epoch):
        losses = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader)
        for iters, (x, y) in enumerate(tqdm_gen):
            x = x.to(self.device)
            y = y.to(self.device)

            loss = self.model(x, y)
            loss_sum = sum(loss.values())

            self.optim.zero_grad()
            loss_sum.backward()
            self.optim.step()
            losses.update(loss_sum.item(), len(y))

            mesg = "epoch {}, loss={:.3f}".format(
                epoch, 
                losses.avg
            )
            tqdm_gen.set_description(mesg)

    def validate(self, dataloader):
        accuracies = []
        acc = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader)
        query_y = torch.arange(self.cfg.n_way).repeat(self.cfg.test.query_per_class_per_episode)
        query_y = query_y.type(torch.LongTensor).to(self.device)
        for episode, batch in enumerate(tqdm_gen):
            batch, _ = [b.to(self.device) for b in batch]
            support_x, query_x = batch[:self.cfg.n_way].unsqueeze(0), batch[self.cfg.n_way:].unsqueeze(0)
            support_y = None
            rewards = self.model.forward_proto(support_x, support_y, query_x, query_y)
            total_rewards = np.sum(rewards)

            accuracy = total_rewards / (query_y.numel())
            acc.update(total_rewards / query_y.numel(), 1)
            mesg = "Val: acc={:.3f}".format(
                acc.avg
            )
            tqdm_gen.set_description(mesg)
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        return test_accuracy, h

    def run(self):
        best_accuracy = 0.0
        best_epoch = -1
        set_seed(1)
        dataloader = make_predataloader(self.cfg, phase="train", batch_size=self.cfg.pre.batch_size)
        val_dataloader = make_predataloader(self.cfg, phase="val")
        for epoch in range(self.epochs):
            self.train(dataloader, epoch + 1)
            self.lr_scheduler.step()

            if epoch > 300 and (epoch + 1) % 5 == 0 or epoch == 0:
                self.model.eval()
                with torch.no_grad():
                    test_accuracy, h = self.validate(val_dataloader)
                mesg = "\t Testing epoch {} validation accuracy: {:.3f}".format(epoch + 1, test_accuracy)
                print(mesg)
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    self.save_model()
                    best_epoch = epoch
                    print("Current best epoch: {}, accuracy: {:.3f}".format(best_epoch + 1, best_accuracy))
                if (epoch + 1) % 100 == 0:
                    self.save_model(postfix=(epoch+1))
                    print("Current best epoch: {}, accuracy: {:.3f}".format(best_epoch + 1, best_accuracy))
                self.model.train()
