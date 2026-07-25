"""HTSG base model."""
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import manifolds
from manifolds.lorentz import Lorentz
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from .hyper_nets import LorentzLinear

class SemanticProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SemanticProjectionLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return F.linear(x, self.weight)

class ResidualSemanticFusion(nn.Module):
    """Residual Semantic Fusion (RSF)"""
    def __init__(self, hidden_size, dropout=0.15):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size * 2, 2, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, txt, img):
        w = torch.softmax(self.gate(torch.cat([self.drop(txt), self.drop(img)], dim=-1)), dim=-1)
        return w[:, 0:1] * txt + w[:, 1:2] * img

class TangentSpaceSemanticGuidance(nn.Module):
    """Hyperbolic tangent-space guidance.
    Applies prototype-guided correction in the Lorentz tangent space after GNN embedding.
    - proto_fake: fake-news directional prototype
    - proto_real: real-news directional prototype
    """
    def __init__(self, lorentz_dim, alpha=0.15, dropout=0.1):
        super().__init__()
        self.lorentz = Lorentz()
        self.proto_fake = nn.Parameter(torch.randn(lorentz_dim) * 0.01)
        self.proto_real = nn.Parameter(torch.randn(lorentz_dim) * 0.01)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        self.drop = nn.Dropout(dropout)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        nn.init.kaiming_normal_(
            self.gate_mlp[0].weight,
            mode="fan_in",
            nonlinearity="relu"
        )
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.zeros_(self.gate_mlp[2].bias)

    def _norm(self, p):
        return p / p.norm().clamp_min(1e-8)

    def forward(self, x_lorentz):
        device = x_lorentz.device
        x_tan = self.lorentz.logmap0(x_lorentz).clamp(-5.0, 5.0)
        x_tan_d = self.drop(x_tan)
        x_n = x_tan_d / x_tan_d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        zero = torch.zeros(1, device=device)
        p_fake = self._norm(torch.cat([zero, self.proto_fake], dim=0))
        p_real = self._norm(torch.cat([zero, self.proto_real], dim=0))
        s_fake = (x_n * p_fake).sum(-1, keepdim=True)
        s_real = (x_n * p_real).sum(-1, keepdim=True)
        gate = self.gate_mlp(torch.cat([s_fake, s_real], dim=-1))
        alpha = self.log_alpha.exp().clamp(0.01, 0.5)
        correction = gate * p_fake.unsqueeze(0) + (1 - gate) * p_real.unsqueeze(0)
        correction = correction / correction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_new = (x_tan + alpha * correction).clamp(-5.0, 5.0)
        out = self.lorentz.expmap0(x_new)
        space = out[:, 1:]
        time = (1.0 + (space * space).sum(-1, keepdim=True).clamp_min(1e-8)).sqrt()
        return torch.cat([time, space], dim=-1)

    def proto_align_loss(self, h, labels):
        """Prototype alignment loss.
        Encourages proto_fake/proto_real to move toward the corresponding class embedding centers.
        h:      [N, dim] Lorentz vectors (after guidance)
        """
        device = h.device
        x_tan = self.lorentz.logmap0(h).clamp(-5.0, 5.0).detach()
        zero = torch.zeros(1, device=device)
        p_fake = torch.cat([zero, self.proto_fake], dim=0)
        p_real = torch.cat([zero, self.proto_real], dim=0)
        mask_fake = (labels == 0)
        mask_real = (labels == 1)
        loss = torch.tensor(0.0, device=device)
        if mask_fake.sum() > 0:
            center_fake = x_tan[mask_fake].mean(0)
            loss = loss + F.mse_loss(p_fake, center_fake.detach())
        if mask_real.sum() > 0:
            center_real = x_tan[mask_real].mean(0)
            loss = loss + F.mse_loss(p_real, center_real.detach())
        return loss

class HTSGBaseModel(nn.Module):
    """
    HTSG model: Residual Semantic Fusion + Tangent-space Semantic Guidance
    """
    def __init__(self, args):
        super(HTSGBaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)
        self.relu = nn.ReLU()
        H = args.hidden_size
        F_DIM = args.feature_dim
        self.text_weight = SemanticProjectionLayer(F_DIM, H)
        self.image_weight = SemanticProjectionLayer(F_DIM, H)
        self.text_liner = LorentzLinear(H + 1, H)
        self.image_liner = LorentzLinear(H + 1, H)
        self.f = LorentzLinear(H + 1, args.dim + 1)
        self.fusion_logit = nn.Parameter(torch.full((H,), 2.1972246))
        self.f_ML = LorentzLinear(args.dim, args.dim)
        self.rsf = ResidualSemanticFusion(H, dropout=0.2)
        self.modal_drop = nn.Dropout(0.1)
        self.tsg = TangentSpaceSemanticGuidance(
            lorentz_dim=args.dim - 1,
            alpha=0.15,
            dropout=0.15
        )

    def encode(self, text_vector, image_vector, adj):
        txt = self.relu(self.text_weight(text_vector))
        img = self.relu(self.image_weight(image_vector))
        txt = self.modal_drop(txt) if self.training else txt
        img = self.modal_drop(img) if self.training else img
        mean_tan = (txt + img) * 0.5
        fused_attn = self.rsf(txt, img)
        fusion_w = torch.sigmoid(self.fusion_logit).unsqueeze(0)
        fused_tan = mean_tan + fusion_w * (fused_attn - mean_tan)
        o = torch.zeros(fused_tan.size(0), 1, device=fused_tan.device)
        fused_hyp = self.manifold.expmap0(torch.cat([o, fused_tan], dim=1))
        x = self.f(fused_hyp)
        h = self.encoder.encode(x, adj)
        h = self.f_ML(h)
        h = self.tsg(h)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class HTSGNodeClassifier(HTSGBaseModel):
    """
    Node classification: proto classifier + proto align loss + gsm_loss
    """
    def __init__(self, args):
        super(HTSGNodeClassifier, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        self.f1_average = 'micro' if args.n_classes > 2 else 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)
        self.n_classes = args.n_classes
        self.n_proto_per_class = 1
        self.proto_tan = nn.Parameter(torch.randn(args.n_classes, self.n_proto_per_class, args.dim - 1) * 0.01)
        self.proto_scale = nn.Parameter(torch.ones(1) * 2.0)
        self.proto_align_w = 0.3
        self.bank_size = int(args.mem_bank_size)
        self.bank_momentum = float(args.mem_bank_momentum)
        self.bank_loss_w = float(args.mem_bank_loss_w)
        self.bank_temp = float(args.mem_bank_temp)
        self.register_buffer('mem_bank', torch.empty(0))
        self.register_buffer('mem_bank_ptr', torch.zeros(self.n_classes, dtype=torch.long))
        self.register_buffer('mem_bank_filled', torch.zeros(self.n_classes, dtype=torch.long))

    def set_class_weights(self, labels, device):
        counts = torch.bincount(labels, minlength=int(labels.max().item()) + 1).float()
        inv_freq = 1.0 / counts.clamp_min(1)
        self.weights = (inv_freq / inv_freq.mean()).to(device)

    def _init_mem_bank(self, dim, device):
        if self.mem_bank.numel() == 0:
            self.mem_bank = torch.zeros(self.n_classes, self.bank_size, dim, device=device)
            self.mem_bank_ptr.zero_()
            self.mem_bank_filled.zero_()

    def _proto_logits(self, h):
        lorentz = Lorentz()
        o = torch.zeros(self.n_classes, self.n_proto_per_class, 1, device=h.device)
        proto = lorentz.expmap0(torch.cat([o, self.proto_tan.to(h.device)], dim=-1))
        h_e = h.unsqueeze(1).unsqueeze(2)          # [N, 1, 1, D]
        p_e = proto.unsqueeze(0)                   # [1, C, P, D]
        mink = (-h_e[..., 0] * p_e[..., 0] + (h_e[..., 1:] * p_e[..., 1:]).sum(-1)).clamp(max=-1.0 - 1e-7)
        dist = torch.acosh(-mink)        
        scores = -dist * self.proto_scale.abs()
        proto_logits = torch.logsumexp(scores, dim=-1)
        return proto_logits, scores

    def _gsm_loss(self, h, labels):
        self._init_mem_bank(h.size(-1), h.device)
        lorentz = Lorentz()
        h_tan = lorentz.logmap0(h).clamp(-5.0, 5.0)
        losses = []
        for c in torch.unique(labels).tolist():
            class_mask = labels == c
            class_h = h_tan[class_mask]
            if class_h.numel() == 0:
                continue
            bank_filled = int(self.mem_bank_filled[c].item())
            if bank_filled == 0:
                continue
            d_k = class_h.mean(dim=0)
            d_k = d_k / d_k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            valid_memories = []
            valid_labels = []
            for q in range(self.n_classes):
                q_filled = int(self.mem_bank_filled[q].item())
                if q_filled == 0:
                    continue
                valid_memories.append(self.mem_bank[q, :q_filled])
                valid_labels.extend([q] * q_filled)
            if not valid_memories:
                continue
            memories = torch.cat(valid_memories, dim=0)
            memories = F.normalize(memories, dim=-1)
            sims = F.cosine_similarity(d_k.unsqueeze(0), memories, dim=-1) / self.bank_temp
            labels_mem = torch.tensor(valid_labels, device=h.device)
            same_mask = labels_mem == c
            if same_mask.sum() == 0:
                continue
            log_num = torch.logsumexp(sims[same_mask], dim=0)
            log_den = torch.logsumexp(sims, dim=0)
            losses.append(-(log_num - log_den))
        if not losses:
            return torch.tensor(0.0, device=h.device)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _update_mem_bank(self, h, labels):
        self._init_mem_bank(h.size(-1), h.device)
        lorentz = Lorentz()
        h_tan = lorentz.logmap0(h).clamp(-5.0, 5.0)
        for c in torch.unique(labels).tolist():
            cls_feat = h_tan[labels == c]
            if cls_feat.numel() == 0:
                continue
            feat = cls_feat.mean(dim=0)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            ptr = int(self.mem_bank_ptr[c].item())
            if self.mem_bank_filled[c] > 0:
                self.mem_bank[c, ptr] = self.bank_momentum * self.mem_bank[c, ptr] + (1.0 - self.bank_momentum) * feat
            else:
                self.mem_bank[c, ptr] = feat
            self.mem_bank[c, ptr] = F.normalize(self.mem_bank[c, ptr], dim=-1)
            self.mem_bank_ptr[c] = (ptr + 1) % self.bank_size
            self.mem_bank_filled[c] = torch.clamp(self.mem_bank_filled[c] + 1, max=self.bank_size)

    def decode(self, h, adj, idx):
        return self._proto_logits(h)[0][idx]

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        labels = data['labels'][idx]
        raw_output = self._proto_logits(embeddings)[0][idx]
        cls_loss = F.cross_entropy(raw_output, labels, weight=self.weights)
        align_loss = self.tsg.proto_align_loss(embeddings[idx], labels)
        gsm_loss = self._gsm_loss(embeddings[idx], labels) if split == 'train' else torch.tensor(0.0, device=embeddings.device)
        loss = cls_loss + self.proto_align_w * align_loss + self.bank_loss_w * gsm_loss
        if split == 'train':
            self._update_mem_bank(embeddings[idx].detach(), labels.detach())
        acc, f1, cr = acc_f1(raw_output, labels, average=self.f1_average)
        return {'loss': loss, 'acc': acc, 'f1': f1}, cr

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["acc"] < m2["acc"]
