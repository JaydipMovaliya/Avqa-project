import torch
import torch.nn as nn
import torch.nn.functional as F

class TrimodalConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, v_feat, a_feat, t_feat):
        """
        v_feat, a_feat, t_feat: (B, D) normalized feature vectors from
        visual, separated audio, and text modalities
        """
        B = v_feat.size(0)

        # Compute similarity matrices (B x B)
        sim_va = torch.matmul(v_feat, a_feat.T) / self.temperature
        sim_at = torch.matmul(a_feat, t_feat.T) / self.temperature
        sim_vt = torch.matmul(v_feat, t_feat.T) / self.temperature

        # Create targets: diagonal = positive pairs
        targets = torch.arange(B).to(v_feat.device)

        # Compute symmetric cross-modal losses
        loss_va = self.cross_entropy(sim_va, targets)
        loss_av = self.cross_entropy(sim_va.T, targets)

        loss_at = self.cross_entropy(sim_at, targets)
        loss_ta = self.cross_entropy(sim_at.T, targets)

        loss_vt = self.cross_entropy(sim_vt, targets)
        loss_tv = self.cross_entropy(sim_vt.T, targets)

        # Combine all 6 directional losses
        total_loss = (loss_va + loss_av + loss_at + loss_ta + loss_vt + loss_tv) / 6.0

        return total_loss