import torch.nn as nn
import torch.nn.functional as F

from model.Abtion import BAFMAbtion
from model.DecisionMLP import DecisionMLP
from model.GruBranch import GruBranch
from model.MambaBranch import MambaBranch
from model.embedding import EmbeddingModel
from model.loss import FocalLoss


class NinaProNetNoBAFM(nn.Module):
    def __init__(self, d_model, num_class, branch_channel, mutil_mode=False, decision_mlp=True, deep_sup=True,
                 cross_subject=True):
        super(NinaProNetNoBAFM, self).__init__()

        if cross_subject:
            self.embedding = EmbeddingModel(branch_channel=branch_channel, embed_hidden_size=16)

        self.mutil_mode = mutil_mode
        self.deep_sup = deep_sup
        self.branch_channel = branch_channel

        self.FFTBranch = GruBranch(d_model=d_model, input_channel=12, output_channel=branch_channel)
        self.TemporalsEMGBranch = MambaBranch(d_model=d_model, input_channel=12, output_channel=branch_channel,
                                              use_BAFM=False)

        self.FeatureFusion = BAFMAbtion(d_model=branch_channel)

        self.final_loss = FocalLoss(in_features=branch_channel, out_features=num_class)

        self.final_out = DecisionMLP(input_size=branch_channel, num_classes=num_class) if decision_mlp else nn.Linear(
            in_features=branch_channel, out_features=num_class)

        self.cross_subject = cross_subject

        self.t_Loss = FocalLoss(in_features=branch_channel, out_features=num_class)
        self.fft_Loss = FocalLoss(in_features=branch_channel, out_features=num_class)

        self.t_out = nn.Linear(in_features=branch_channel, out_features=num_class)
        self.fft_out = nn.Linear(in_features=branch_channel, out_features=num_class)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, s_emg, acc, labels, visual=False, subject_feature=None, return_logits=False):
        s_emg = s_emg.float()

        temporal_features = self.TemporalsEMGBranch(s_emg)
        fft_features = self.FFTBranch(s_emg)

        if self.cross_subject:
            categorical_data, continuous_data = subject_feature
            subject_feature = self.embedding(categorical_data, continuous_data)
        else:
            subject_feature = None

        feature = self.FeatureFusion(temporal_features, fft_features, subject_feature)
        feature = self.max_pool(feature).squeeze(-1)
        feature = F.dropout(feature, p=0.2, training=self.training)
        out = self.final_out(feature)

        visual_feature = {
            "temporal_features": F.normalize(temporal_features, p=2, dim=-1),
            "fft_features": F.normalize(fft_features, p=2, dim=-1),
            "fusion_feature": F.normalize(feature, p=2, dim=-1),
            "out": F.normalize(out, p=2, dim=-1)
        }

        temporal_out = self.t_out(temporal_features.mean(1))
        fft_out = self.fft_out(fft_features.mean(1))

        t_Loss, t_acc = self.t_Loss(temporal_out, labels)
        fft_Loss, fft_acc = self.fft_Loss(fft_out, labels)

        final_loss, final_accuracy = self.final_loss(out, labels)

        loss = final_loss + t_Loss + fft_Loss if self.deep_sup else final_loss

        if visual:
            return loss, final_accuracy, visual_feature

        if return_logits:
            return out

        return loss, final_accuracy, t_acc, fft_acc
