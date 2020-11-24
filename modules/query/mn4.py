import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils import batched_index_select
from .innerproduct_similarity import InnerproductSimilarity

@registry.Query.register("MN4")
class MN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.inner_simi = InnerproductSimilarity(cfg, metric='cosine')
        self.nbnn_topk = cfg.model.nbnn_topk
        self.temperature = cfg.model.temperature
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg

    def _MNN(self, simi_matrix, compensate_for_single):
        b, q, N, M_q, M_s = simi_matrix.shape
        simi_matrix_merged = simi_matrix.permute(0, 1, 3, 2, 4).contiguous().view(b, q, M_q, -1)
        query_nearest = simi_matrix_merged.max(-1)[1]
        if not compensate_for_single:
            support_nearest = simi_matrix_merged.max(-2)[1] # For old Conv4 version
        else:
            class_wise_max = (simi_matrix.max(-1)[0]).max(2)[0] + 1
            class_m = torch.nn.functional.one_hot(query_nearest, self.n_way * M_s).float() * class_wise_max.unsqueeze(-1)
            class_m_max, support_nearest = class_m.max(-2)
        # [b, q, M_q]
        mask = batched_index_select(support_nearest.view(-1, self.n_way * M_s).unsqueeze(1), 2, query_nearest.view(-1, M_q))
        mask = (mask == torch.arange(M_q, device=simi_matrix.device).expand_as(mask)).view(b, q, M_q)
        return mask

    def forward(self, support_xf, support_y, query_xf, query_y):
        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]
        simi_matrix = self.inner_simi(support_xf, support_y, query_xf, query_y)
        q_mask = self._MNN(simi_matrix, self.cfg.model.encoder != "FourLayer_64F")
        query_value = (torch.topk(simi_matrix, self.nbnn_topk, -1)[0].mean(-1) * q_mask.float().unsqueeze(2)).sum(-1)
        query_value = query_value.view(b*q, self.n_way)
        query_y = query_y.view(b * q)
        if self.training:
            loss = self.criterion(query_value / self.temperature, query_y)
            return {"MN4_loss": loss}
        else:
            _, predict_labels = torch.max(query_value, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
