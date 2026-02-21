import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, memory_size=2, feature_dim=64, key_dim=64, temp_update=0.1, temp_gather=0.1, momentum=0.9, num_prototypes=4, update_interval=10, topk_queries=0.1, max_skip_batches=5):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        self.momentum = momentum
        self.num_prototypes = num_prototypes
        self.update_interval = update_interval
        self.topk_queries = topk_queries  # Now a percentage (e.g., 0.1 for 10%)
        self.max_skip_batches = max_skip_batches

        self.register_buffer('ltm', F.normalize(torch.randn(memory_size, num_prototypes, feature_dim), dim=-1))
        self.register_buffer('stm', torch.zeros(memory_size, num_prototypes, feature_dim))
        self.register_buffer('update_counts', torch.zeros(memory_size, num_prototypes, dtype=torch.long))
        self.register_buffer('skip_counts', torch.zeros(memory_size, dtype=torch.long))
        self.query_proj = nn.Sequential(
            nn.Conv2d(feature_dim, key_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_dim),
            nn.ReLU()
        )
        self.attn_temp = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels=None, train=True):
        B, C, H, W = x.shape
        x_norm = F.normalize(x, dim=1)
        query = self.query_proj(x_norm)
        query = query.view(B, self.key_dim, -1).permute(0, 2, 1)
        query = F.normalize(query, dim=-1)

        ltm_norm = F.normalize(self.ltm.detach(), dim=-1).contiguous()
        stm_norm = F.normalize(self.stm, dim=-1).contiguous()
        memory = 0.7 * ltm_norm + 0.3 * stm_norm
        memory_flat = memory.reshape(self.memory_size * self.num_prototypes, self.feature_dim).contiguous()
        sim = torch.einsum('bnc,mc->bnm', query, memory_flat) / self.attn_temp
        attn = F.softmax(sim, dim=-1)
        memory_features = memory_flat.t()
        retrieved = torch.einsum('bnm,cm->bnc', attn, memory_features)
        retrieved = retrieved.view(B, H, W, self.feature_dim).permute(0, 3, 1, 2)
        # updated_x = torch.cat([x, retrieved], dim=1)
        updated_x = x + retrieved

        if train:
            self._update_memory(x_norm, sim, labels)
        return updated_x

    def _update_memory(self, x_norm, sim, labels):
        B, C, H, W = x_norm.shape
        x_flat = x_norm.view(B, self.feature_dim, -1).permute(0, 2, 1)
        labels_flat = labels.view(B, -1)
        sim_per_class = sim.view(B, H*W, self.memory_size, self.num_prototypes)

        for cls in range(self.memory_size):
            cls_mask = (labels_flat == cls)
            cls_sim = sim_per_class[:, :, cls, :]
            cls_sim_flat = cls_sim.view(B * H * W, self.num_prototypes)
            cls_mask_flat = cls_mask.view(B * H * W)


            if cls_mask.sum() > 0:
                self.skip_counts[cls] = 0
                valid_sim = cls_sim_flat[cls_mask_flat]
                valid_pixels = x_flat.reshape(B * H * W, self.feature_dim)[cls_mask_flat]
                k = max(1, min(int(cls_mask.sum() * self.topk_queries), 50))  # Dynamic topk
            else:
                self.skip_counts[cls] += 1
                if self.skip_counts[cls] < self.max_skip_batches:
                    continue
                valid_sim = cls_sim_flat
                valid_pixels = x_flat.reshape(B * H * W, self.feature_dim)
                k = max(1, min(int((H * W) * self.topk_queries), 50))  # Fallback topk
                self.skip_counts[cls] = 0

            for proto in range(self.num_prototypes):
                proto_sim = valid_sim[:, proto]
                topk_scores, topk_indices = torch.topk(proto_sim, k, dim=0)
                topk_feats = valid_pixels[topk_indices]
                topk_feats = F.normalize(topk_feats, dim=-1)
                weights = F.softmax(topk_scores / self.temp_update, dim=0)
                if cls_mask.sum() == 0:
                    weights *= 0.1
                avg_feat = (topk_feats * weights.unsqueeze(-1)).sum(dim=0)
                self.stm[cls, proto] = self.momentum * self.stm[cls, proto] + \
                                      (1 - self.momentum) * avg_feat.detach()
                self.stm[cls, proto] = F.normalize(self.stm[cls, proto], dim=-1)
                self.update_counts[cls, proto] += 1

        if self.update_counts.max() >= self.update_interval:
            for cls in range(self.memory_size):
                for proto in range(self.num_prototypes):
                    if self.update_counts[cls, proto] >= self.update_interval:
                        stm_vec = self.stm[cls, proto].unsqueeze(0)
                        ltm_vec = self.ltm[cls, proto].unsqueeze(0)
                        gate_input = torch.cat([stm_vec, ltm_vec], dim=-1)
                        alpha = self.gate(gate_input).squeeze()
                        updated_ltm = (1 - alpha) * self.ltm[cls, proto] + alpha * self.stm[cls, proto]
                        self.ltm[cls, proto] = updated_ltm.detach().contiguous()
                        self.stm[cls, proto] = torch.zeros_like(self.stm[cls, proto]).detach()
                        self.update_counts[cls, proto] = 0