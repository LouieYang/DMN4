import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

from modules.utils import _l2norm
from modules.query.innerproduct_similarity import InnerproductSimilarity

def batched_index_select(input_, dim, index):
    for ii in range(1, len(input_.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input_.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input_, dim, index)

@registry.SemiQuery.register("MN4")
class MN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.inner_simi = InnerproductSimilarity(cfg, metric='cosine')
        self.temperature = cfg.model.temperature
        self.criterion = nn.CrossEntropyLoss()

        self.backbone = cfg.model.encoder

    def forward(self, support_xf, support_y, query_xf, query_y, unlabeled_xf):
        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]
        unlabeled_xf = unlabeled_xf.view(b, -1, c, h, w)
        u = unlabeled_xf.shape[1]
        unlabeled_xf_pool = unlabeled_xf.permute(0, 2, 1, 3, 4).contiguous().view(b, 1, c, u * h, w)

        u2s = self.inner_simi(support_xf, None, unlabeled_xf_pool, None) # [b, 1, N, M_u, M_s], M_u = u * h * w
        M_u, M_s = u2s.shape[-2:]
        u2s_pool = u2s.permute(0, 1, 3, 2, 4).contiguous().view(b, 1, M_u, -1)
        s_nearest = u2s_pool.max(-2)[1]
        u_nearest = u2s_pool.max(-1)[1]
        u2s_mask = batched_index_select(s_nearest.view(-1, self.n_way * M_s).unsqueeze(1), 2, u_nearest.view(-1, M_u))
        u2s_mask = (u2s_mask == torch.arange(M_u, device=device).expand_as(u2s_mask)).view(b, 1, M_u)
        # scatter: dispatch each unlabeled descriptors to support class via DualNN
        umask = u2s.max(-1)[0] # [b, 1, N, M_u]
        umask = umask.transpose(-2, -1).max(-1)[1] # [b, 1, M_u]
        umask = torch.nn.functional.one_hot(umask, self.n_way).float() # [b, 1, M_u, N]
        umask = umask.transpose(-2, -1) * u2s_mask.unsqueeze(2).float() # [b, 1, N, M_u]
        umask = umask.squeeze(1)

        umask_length = umask.sum(-1).max().long()
        unlabeled_dualnned_tensor = torch.zeros((b, self.n_way, umask_length, c), device=device)
        unlabeled_xf_pool = unlabeled_xf_pool.view(b, c, M_u).transpose(-2, -1) # [b, M_u, c]
        for i, (umask_per_batch, unlabeled_xf_per_batch) in enumerate(zip(umask, unlabeled_xf_pool)):
            # umask_per_batch: [N, M_u]
            # unlabeled_xf_per_batch: [M_u, c]
            unlabeled_scatter_per_batch = []
            umask_classes = torch.split(umask_per_batch, 1, dim=0)
            for j, umask_per_class in enumerate(umask_classes):
                umask_per_class = umask_per_class.squeeze(0)
                unlabeled_dualnned_tensor[i, j, :umask_per_class.sum().long(), :] = unlabeled_xf_per_batch[umask_per_class.byte(), :]
        unlabeled_dualnned_tensor = unlabeled_dualnned_tensor.transpose(-2, -1) # [b, self.n_ways, c, ?]
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).permute(0, 1, 3, 2, 4, 5)
        support_xf = support_xf.contiguous().view(b, self.n_way, c, -1)
        s_cat_u = torch.cat((support_xf, unlabeled_dualnned_tensor), dim=-1)

        # MN4 forward
        query_matrix_ori = self.inner_simi(s_cat_u, None, query_xf, None) # [b, q, N, M_q, M_s]
        if self.backbone == "FourLayer_64F":
            query_matrix_ori = (query_matrix_ori + 1) / 2
        M_q, M_s = query_matrix_ori.shape[-2:]
        query_matrix = query_matrix_ori.permute(0, 1, 3, 2, 4).contiguous().view(b, q, M_q, -1)
        support_nearest = query_matrix.max(-2)[1]
        query_value, query_nearest = query_matrix.max(-1)

        # q_mask: [b, q, M_q]
        # s_mask: [b, q, M_s]
        q_mask = batched_index_select(support_nearest.view(-1, self.n_way * M_s).unsqueeze(1), 2, query_nearest.view(-1, M_q))
        q_mask = (q_mask == torch.arange(M_q, device=device).expand_as(q_mask)).view(b, q, M_q)

        query_value = (query_matrix_ori.max(-1)[0] * q_mask.float().unsqueeze(2)).sum(-1)
        query_value = query_value.view(b*q, self.n_way)

        query_y = query_y.view(b * q)
        if self.training:
            loss = self.criterion(query_value / self.temperature, query_y)
            return {"MN4_loss": loss}
        else:
            _, predict_labels = torch.max(query_value, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards

