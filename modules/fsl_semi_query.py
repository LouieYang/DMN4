import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import make_encoder
from .semi_query import make_query

class FSLSemiQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = make_encoder(cfg)
        self.query = make_query(self.encoder.out_channels, cfg)

    def forward(self, support_x, support_y, query_x, query_y, unlabeled_x):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]

        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        unlabeled_xf = self.encoder(unlabeled_x.view(-1, c, h, w))

        fc, fh, fw = support_xf.shape[-3:]
        support_xf = support_xf.view(b, s, fc, fh, fw)
        query_xf = query_xf.view(b, q, fc, fh, fw)

        query = self.query(support_xf, support_y, query_xf, query_y, unlabeled_xf)
        return query

def make_semi_fsl(cfg):
    return FSLSemiQuery(cfg)

