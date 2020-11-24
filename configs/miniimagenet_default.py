import os.path as osp
from .collections import AttrDict

cfg = AttrDict()

cfg.model = AttrDict()
cfg.model.encoder = "FourLayer_64F"
cfg.model.pyramid_list = []
cfg.model.query = "DN4"
cfg.model.mnn_k = 1 # deprecated
cfg.model.nbnn_topk = 1
cfg.model.temperature = 1.0

cfg.n_way = 5
cfg.k_shot = 5

cfg.train = AttrDict()
cfg.train.query_per_class_per_episode = 10
cfg.train.episode_per_epoch = 20000
cfg.train.epochs = 20
cfg.train.colorjitter = False
cfg.train.learning_rate = 0.001
cfg.train.lr_decay = 0.1 
cfg.train.lr_decay_epoch = 10
cfg.train.lr_decay_milestones = []
cfg.train.adam_betas = (0.5, 0.9)
cfg.train.sgd_mom = 0.9
cfg.train.optim = "Adam"
cfg.train.sgd_weight_decay = 5e-4
cfg.train.batch_size = 4
cfg.train.fix_bn = False # only for WRN-28-10 backbone

cfg.val = AttrDict()
cfg.val.episode = 1000

cfg.test = AttrDict()
cfg.test.query_per_class_per_episode = 15
cfg.test.episode = 600
cfg.test.total_testtimes = 10
cfg.test.batch_size = 4

cfg.data = AttrDict()
cfg.data.root = "./data/"
cfg.data.image_dir = "./data/miniImagenet"

cfg.pre = AttrDict()
cfg.pre.lr = 0.1
cfg.pre.lr_decay = 0.1
cfg.pre.lr_decay_milestones = [100, 200, 250, 300]
cfg.pre.epochs = 350
cfg.pre.batch_size = 128
cfg.pre.colorjitter = True
cfg.pre.val_episode = 200
cfg.pre.pretrain_num_class = 64

# For semi-fsl
cfg.semi = AttrDict()
cfg.semi.label_percentage = 0.4
cfg.semi.with_distractors = 0 # default w/o D
cfg.semi.distractor_class = 5 # H
cfg.semi.unlabeled_per_task_train = 5 # M_train
cfg.semi.unlabeled_per_task_test = 20 # M_test
