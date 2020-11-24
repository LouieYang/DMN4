import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from modules.utils import batched_index_select
from .innerproduct_similarity import InnerproductSimilarity

@registry.Query.register("DMN4")
class DMN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.inner_simi = InnerproductSimilarity(cfg, metric='cosine')
        self.temperature = cfg.model.temperature
        self.criterion = nn.CrossEntropyLoss()

    def _DMNN(self, simi_matrix):
        b, q, N, M_q, M_s = simi_matrix.shape
        simi_matrix_merged = simi_matrix.permute(0, 1, 3, 2, 4).contiguous().view(b, q, M_q, -1)

        query_nearest = simi_matrix_merged.max(-1)[1]
        query_class_diff = torch.topk(simi_matrix.max(-1)[0], 2, 2)[0]
        query_class_diff = query_class_diff[:, :, 0, :] - query_class_diff[:, :, 1, :] # [b, q, M_q]
        diffs_m = torch.nn.functional.one_hot(query_nearest, self.n_way * M_s).float() * query_class_diff.unsqueeze(-1)
        diffs_m_max, diffs_max_nearest = diffs_m.max(-2)

        diff_mask = batched_index_select(diffs_max_nearest.view(-1, self.n_way * M_s).unsqueeze(1), 2, query_nearest.view(-1, M_q))
        diff_mask = (diff_mask == torch.arange(M_q, device=simi_matrix.device).expand_as(diff_mask)).view(b, q, M_q)
        return diff_mask

    def forward(self, support_xf, support_y, query_xf, query_y):
        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]
        simi_matrix = self.inner_simi(support_xf, support_y, query_xf, query_y)
        q_mask = self._DMNN(simi_matrix)

        query_value = (simi_matrix.max(-1)[0] * q_mask.float().unsqueeze(2)).sum(-1)
        query_value = query_value.view(b*q, self.n_way)
        query_y = query_y.view(b * q)
        if self.training:
            loss = self.criterion(query_value / self.temperature, query_y)
            return {"DMN4_loss": loss}
        else:
            _, predict_labels = torch.max(query_value, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards
