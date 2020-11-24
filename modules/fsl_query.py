import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import make_encoder
from .query import make_query

class FSLQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = make_encoder(cfg)
        self.query = make_query(self.encoder.out_channels, cfg)
        self.pyramid_list = cfg.model.pyramid_list

    def pyramid_encoding(self, x):
        b, n, c, h , w = x.shape
        x = x.view(-1, c, h, w)
        feature_list = []
        for size in self.pyramid_list:
            feature_list.append(F.adaptive_avg_pool2d(x, size).view(b, n, c, 1, -1))
        feature_list.append(x.view(b, n, c, 1, -1))
        out = torch.cat(feature_list, dim=-1)
        out = out - out.mean(2).unsqueeze(2)
        return out

    def forward(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]

        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        fc, fh, fw = support_xf.shape[-3:]
        support_xf = support_xf.view(b, s, fc, fh, fw)
        query_xf = query_xf.view(b, q, fc, fh, fw)

        if self.pyramid_list:
            support_xf = self.pyramid_encoding(support_xf)
            query_xf = self.pyramid_encoding(query_xf)

        query = self.query(support_xf, support_y, query_xf, query_y)
        return query

def make_fsl(cfg):
    return FSLQuery(cfg)
