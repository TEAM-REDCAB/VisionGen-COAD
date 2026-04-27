# fusion 부분 마스킹한 파일 참조하도록 수정

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from modules.path_encoder    import PathEncoder
from modules.genomic_encoder import GenomicEncoder
from modules.coattn_fusion_v3_1   import CoAttnFusion
from modules.classifier_head import ClassifierHead


class MCATTeacher(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict,
        path_dim:   int   = 1536,
        n_classes:  int   = 2,
        dropout:    float = 0.5,
        mcat_path:  str   = './mcat',
    ):
        super().__init__()
        print("[MCATTeacher] 모듈 조립 완료: PathEncoder → GenomicEncoder → CoAttnFusion → ClassifierHead")

        self.path_encoder    = PathEncoder(in_dim=path_dim, out_dim=256, dropout=dropout)
        self.genomic_encoder = GenomicEncoder(vocab_sizes=vocab_sizes, out_dim=256, dropout=dropout)
        self.coattn_fusion   = CoAttnFusion(embed_dim=256, num_heads=1, dropout=dropout, mcat_path=mcat_path)
        self.classifier      = ClassifierHead(in_dim=256, n_classes=n_classes)

    def forward(self, x_path, x_omic):
        """
        x_path : (N_patches, 1536)
        x_omic : (1425, 9) 또는 (1, 1425, 9)
        """
        # dataset.py가 (1, 1425, 9)로 넘길 수 있으므로 squeeze
        if x_omic.dim() == 3 and x_omic.shape[0] == 1:
            x_omic = x_omic.squeeze(0)

        # PathEncoder는 N_patches가 클 수 있어 grad_checkpoint 적용
        h_path = grad_checkpoint(self.path_encoder, x_path, use_reentrant=False)  # (N, 256)
        h_omic = self.genomic_encoder(x_omic)                                      # (1425, 256)

        h_fused, attn_scores = self.coattn_fusion(h_path, h_omic)                 # (256,), dict

        logits = self.classifier(h_fused)                                          # (1, 2)
        Y_hat  = torch.topk(logits, 1, dim=1)[1]                                  # (1, 1)

        return logits, Y_hat, attn_scores
