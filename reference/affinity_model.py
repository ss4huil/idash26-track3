"""
Plaintext affinity reference model (spec §4, §5).

Combines the three pieces into an end-to-end DTA predictor that mirrors
DeepDTAGen's affinity path but uses the dense fixed-Nmax GCN (the form that
gets secret-shared into MPC):

    drug_path    : GCN×3(+ReLU) → masked max-pool → Drug_FC(→1024→ReLU→128)  [MPC]
    protein_path : GatedCNN → Pvec(128)                                       [plaintext/public]
    predict      : concat(PMVO, Pvec) → FC(256→1024→512→256→1)                [MPC]

This is the golden reference the MPC C++ implementation must reproduce.
The GatedCNN is reimplemented standalone (no fairseq dependency) so it can
run without the generation stack.
"""
import numpy as np
import torch
import torch.nn as nn

# ── protein sequence encoding (matches DeepDTAGen/create_data.py) ─────────────
_SEQ_VOC     = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
_SEQ_DICT    = {v: (i + 1) for i, v in enumerate(_SEQ_VOC)}
_MAX_SEQ_LEN = 1000


def seq_cat(prot: str) -> np.ndarray:
    """Encode a protein sequence into a length-1000 integer vector."""
    x = np.zeros(_MAX_SEQ_LEN, dtype=np.int64)
    for i, ch in enumerate(prot[:_MAX_SEQ_LEN]):
        x[i] = _SEQ_DICT.get(ch, 0)
    return x


class GatedCNN(nn.Module):
    """Plaintext protein encoder — public, standalone (no fairseq)."""

    def __init__(self, protein_features=25, num_filters=32,
                 embed_dim=128, final_dim=128, k_size=8):
        super().__init__()
        self.Protein_Embed = nn.Embedding(protein_features + 1, embed_dim)
        self.Protein_Conv1 = nn.Conv1d(1000, num_filters, k_size)
        self.Protein_Gate1 = nn.Conv1d(1000, num_filters, k_size)
        self.Protein_Conv2 = nn.Conv1d(num_filters, num_filters * 2, k_size)
        self.Protein_Gate2 = nn.Conv1d(num_filters, num_filters * 2, k_size)
        self.Protein_Conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, k_size)
        self.Protein_Gate3 = nn.Conv1d(num_filters * 2, num_filters * 3, k_size)
        self.relu = nn.ReLU()
        self.Protein_FC = nn.Linear(96 * 107, final_dim)

    def forward(self, target):                       # target: (B, 1000) long
        e = self.Protein_Embed(target)               # (B, 1000, 128)
        o1 = self.relu(self.Protein_Conv1(e) * torch.sigmoid(self.Protein_Gate1(e)))
        o2 = self.relu(self.Protein_Conv2(o1) * torch.sigmoid(self.Protein_Gate2(o1)))
        o3 = self.relu(self.Protein_Conv3(o2) * torch.sigmoid(self.Protein_Gate3(o2)))
        xt = o3.reshape(o3.size(0), -1)              # (B, 96*107)
        return self.Protein_FC(xt)                    # (B, 128)


def _relu(x):
    return np.maximum(x, 0)


class AffinityModel:
    """End-to-end plaintext affinity predictor (dense GCN form)."""

    def __init__(self, gcn, drug_fc, gated_cnn, fusion, feat_dim=94):
        self.gcn     = gcn        # [(W,b)]×3 GCN layers
        self.drug_fc = drug_fc    # [(W,b)]×2 Drug_FCs (Linear, Linear)
        self.gated   = gated_cnn  # GatedCNN torch module (eval)
        self.fusion  = fusion     # [(W,b)]×4 fusion FC layers
        self.feat_dim = feat_dim

    # ── constructors ──────────────────────────────────────────────────────────
    @classmethod
    def from_random(cls, feat_dim=94, seed=0):
        rng = np.random.default_rng(seed)

        def lin(o, i):
            return (rng.standard_normal((o, i)).astype(np.float32) * 0.1,
                    rng.standard_normal(o).astype(np.float32) * 0.1)

        gcn     = [lin(188, feat_dim), lin(282, 188), lin(376, 282)]
        drug_fc = [lin(1024, 376), lin(128, 1024)]
        fusion  = [lin(1024, 256), lin(512, 1024), lin(256, 512), lin(1, 256)]
        gated   = GatedCNN().eval()
        return cls(gcn, drug_fc, gated, fusion, feat_dim)

    @classmethod
    def from_pth(cls, pth_path, tokenizer_path=None, device="cpu"):
        sd = torch.load(pth_path, map_location=device)

        def _wb(prefix):
            # GCNConv weight may live under '.lin.weight' (newer PyG) or '.weight'
            wk = f"{prefix}.lin.weight" if f"{prefix}.lin.weight" in sd else f"{prefix}.weight"
            W = sd[wk].cpu().numpy()
            if wk.endswith(".weight") and ".lin." not in wk and W.shape[0] != _bias_len(prefix):
                # older PyG stores GCN weight as (in, out) → transpose to (out, in)
                W = W.T
            b = sd[f"{prefix}.bias"].cpu().numpy()
            return W.astype(np.float32), b.astype(np.float32)

        def _bias_len(prefix):
            return sd[f"{prefix}.bias"].shape[0]

        gcn = [_wb("encoder.GraphConv1"),
               _wb("encoder.GraphConv2"),
               _wb("encoder.GraphConv3")]
        drug_fc = [_wb("encoder.Drug_FCs.0"), _wb("encoder.Drug_FCs.3")]
        fusion  = [_wb("fc.FC_layers.0"), _wb("fc.FC_layers.3"),
                   _wb("fc.FC_layers.6"), _wb("fc.FC_layers.9")]

        # GatedCNN: load the 'cnn.' subset
        gated = GatedCNN()
        cnn_sd = {k[len("cnn."):]: v for k, v in sd.items() if k.startswith("cnn.")}
        gated.load_state_dict(cnn_sd)
        gated.eval()

        feat_dim = gcn[0][0].shape[1]
        return cls(gcn, drug_fc, gated, fusion, feat_dim)

    # ── forward pieces ──────────────────────────────────────────────────────────
    def drug_path(self, X, A_hat, mask):
        from reference.dense_gcn     import dense_gcn_layer
        from reference.masked_maxpool import masked_global_max_pool
        H = X
        for (W, b) in self.gcn:
            H = _relu(dense_gcn_layer(H, A_hat, W, b))
        pooled = masked_global_max_pool(H, mask)         # (376,)
        W0, b0 = self.drug_fc[0]
        h = _relu(pooled @ W0.T + b0)                    # (1024,)
        W1, b1 = self.drug_fc[1]
        return h @ W1.T + b1                              # (128,) PMVO

    def protein_path(self, protein_seq):
        enc = torch.tensor(seq_cat(protein_seq), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            return self.gated(enc).numpy().squeeze(0)     # (128,) Pvec

    def predict(self, X, A_hat, mask, protein_seq):
        pmvo = self.drug_path(X, A_hat, mask)             # (128,)
        pvec = self.protein_path(protein_seq)             # (128,)
        h = np.concatenate([pmvo, pvec])                  # (256,) drug first
        n = len(self.fusion)
        for k, (W, b) in enumerate(self.fusion):
            h = h @ W.T + b
            if k < n - 1:
                h = _relu(h)                              # ReLU on all but last
        return float(h[0])

    def predict_batch(self, pairs, nmax=128):
        from reference.dense_graph import smile_to_dense_graph
        out = []
        for smile, protein in pairs:
            X, A_hat, mask = smile_to_dense_graph(smile, nmax)
            out.append(self.predict(X, A_hat, mask, protein))
        return np.array(out, dtype=np.float32)
