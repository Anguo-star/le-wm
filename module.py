import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class WassersteinSIGReg(torch.nn.Module):
    """Sliced-Wasserstein Gaussian regularizer (scale-aware SIGReg warmup).

    Shares SIGReg's input convention (``proj: (T, B, D)``). Each random 1D
    projection of the embeddings is compared against standard-normal quantiles
    with the squared 2-Wasserstein distance. Unlike the Epps-Pulley ``SIGReg``,
    whose fixed knots assume the input already sits near unit scale, this loss
    has a scale-aware gradient that actively pulls each projected marginal onto
    ``N(0, 1)`` regardless of its current scale. It is meant as a warmup that
    fixes embedding scale so the Epps-Pulley SIGReg's knots become well-aligned
    afterwards. It is disabled by default.
    """

    def __init__(self, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        proj = proj.float()
        # Treat the batch dimension as the sample dimension per time step (B
        # samples), matching SIGReg which averages its statistic over B only.
        sample_count = proj.size(1)
        if sample_count < 2:
            return proj.new_tensor(0.0)
        # sample random projections (unit-norm directions)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0).clamp_min(torch.finfo(proj.dtype).eps))
        # (T, B, num_proj): sort each projected marginal over B and match it to
        # standard-normal quantiles; the final mean averages over T and num_proj.
        projected = (proj @ A).sort(dim=1).values
        quantile_positions = (
            torch.arange(sample_count, device=proj.device, dtype=proj.dtype) + 0.5
        ) / float(sample_count)
        normal_quantiles = torch.erfinv(2.0 * quantile_positions - 1.0) * (2.0**0.5)
        return (projected - normal_quantiles.view(1, sample_count, 1)).square().mean()


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


def transition_pair_features(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    *,
    detach: bool = False,
) -> torch.Tensor:
    """Concatenate consecutive latent pairs for transition-level heads."""
    if z_t.shape != z_tp1.shape:
        raise ValueError(
            f"z_t and z_tp1 must have the same shape, got {z_t.shape} vs {z_tp1.shape}"
        )
    if detach:
        z_t = z_t.detach()
        z_tp1 = z_tp1.detach()
    return torch.cat([z_t, z_tp1], dim=-1)


def inverse_dynamics_loss(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    action: torch.Tensor,
    head: nn.Module,
    *,
    detach_input: bool = False,
) -> torch.Tensor:
    """Predict the action that connects consecutive latent states."""
    if z_t.numel() == 0:
        return z_t.new_tensor(0.0)
    pred_action = head(transition_pair_features(z_t, z_tp1, detach=detach_input))
    if pred_action.shape != action.shape:
        raise ValueError(
            "inverse dynamics prediction and action target must have the same "
            f"shape, got {pred_action.shape} vs {action.shape}"
        )
    return F.mse_loss(pred_action, action.float())


def transition_distance_prediction_loss(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    head: nn.Module,
    *,
    metric: str,
    detach_input: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict consecutive-state latent distance as a diagnostic auxiliary."""
    if z_t.numel() == 0:
        empty = z_t.new_empty(z_t.shape[:-1])
        return z_t.new_tensor(0.0), empty, empty

    pred_dist = F.softplus(
        head(transition_pair_features(z_t, z_tp1, detach=detach_input)).squeeze(-1)
    )

    if metric == "l2":
        target_dist = torch.linalg.vector_norm(z_tp1 - z_t, dim=-1)
    elif metric == "cosine":
        z_t_norm = F.normalize(z_t, dim=-1, eps=1e-8)
        z_tp1_norm = F.normalize(z_tp1, dim=-1, eps=1e-8)
        target_dist = 1.0 - (z_t_norm * z_tp1_norm).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported transition distance metric: {metric}")

    loss = F.smooth_l1_loss(pred_dist, target_dist.detach())
    return loss, pred_dist, target_dist


def cosine_pred_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Prediction loss on the sphere: mean cosine distance.

    pred, target: (B, T, D) — unit vectors on S^{d-1}.
    Returns scalar in [0, 2]; 0 means perfect alignment.
    """
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def temporal_hinge_loss(
    z_t: torch.Tensor,
    z_tp1: torch.Tensor,
    *,
    margin: float,
    metric: str,
    squared: bool = True,
) -> torch.Tensor:
    """Upper hinge loss on consecutive latent pairs.

    z_t, z_tp1: (..., D) consecutive latent embeddings with matching shapes.
    metric:
      - "l2": Euclidean distance in R^d
      - "cosine": cosine distance 1 - cos(z_t, z_tp1), suitable for unit vectors

    This encourages consecutive latents to stay close, but stops rewarding
    them once their distance is already below the margin:
      max(0, distance(z_t, z_{t+1}) - margin)
    """
    if z_t.shape != z_tp1.shape:
        raise ValueError(
            f"z_t and z_tp1 must have the same shape, got {z_t.shape} vs {z_tp1.shape}"
        )
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    if metric == "l2":
        dist = torch.linalg.vector_norm(z_tp1 - z_t, dim=-1)
    elif metric == "cosine":
        z_t = F.normalize(z_t, dim=-1, eps=1e-8)
        z_tp1 = F.normalize(z_tp1, dim=-1, eps=1e-8)
        dist = 1.0 - (z_t * z_tp1).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported temporal hinge metric: {metric}")

    loss = torch.clamp_min(dist - margin, 0.0)
    if squared:
        loss = loss.square()
    return loss.mean()


def temporal_straightness(z: torch.Tensor) -> torch.Tensor:
    """Mean cosine between consecutive latent displacement vectors.

    z: (B, T, D) encoder outputs on real trajectories.
    Returns a scalar; larger values mean temporally straighter trajectories.
    """
    if z.ndim != 3:
        raise ValueError(f"z must have shape (B, T, D), got {z.shape}")
    if z.size(1) < 3:
        return z.new_tensor(0.0)

    v = z[:, 1:] - z[:, :-1]  # (B, T-1, D)
    v1 = v[:, :-1]  # (B, T-2, D)
    v2 = v[:, 1:]  # (B, T-2, D)

    denom = v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8
    cos = (v1 * v2).sum(dim=-1) / denom
    return cos.mean()


def _pairwise_offdiag(x: torch.Tensor) -> torch.Tensor:
    """Return all off-diagonal pairwise entries for a flattened batch."""
    n = x.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return x[mask]


def spread_loss(emb: torch.Tensor, margin: float) -> torch.Tensor:
    """Anti-collapse loss: mean squared hinge on pairwise cosine similarity.

    emb: (B, T, D) — unit vectors on S^{d-1}.
    Computes mean_{i != j} max(0, <mu_i, mu_j> - m)^2.
    Minimising this only penalises pairs whose cosine similarity exceeds the
    configured margin.
    """
    z = emb.reshape(-1, emb.size(-1))  # (B*T, D)
    sim = z @ z.T  # (B*T, B*T)
    return torch.clamp_min(_pairwise_offdiag(sim) - margin, 0.0).square().mean()


def _uniformity_pair_mask(
    batch_size: int,
    seq_len: int,
    *,
    mode: str,
    temporal_exclusion: int,
    device,
) -> torch.Tensor:
    """Return the pair mask used by uniformity regularization."""
    if temporal_exclusion < 0:
        raise ValueError("uniformity temporal_exclusion must be >= 0")

    n = batch_size * seq_len
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    if mode == "all_pairs":
        return upper

    batch_ids = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
    if mode == "cross_window":
        return upper & (batch_ids[:, None] != batch_ids[None, :])

    if mode == "temporal_masked":
        time_ids = torch.arange(seq_len, device=device).repeat(batch_size)
        near_temporal = (batch_ids[:, None] == batch_ids[None, :]) & (
            (time_ids[:, None] - time_ids[None, :]).abs() <= temporal_exclusion
        )
        return upper & ~near_temporal

    raise ValueError(f"Unsupported uniformity.mode: {mode}")


def uniformity_loss(
    emb: torch.Tensor,
    t: float,
    *,
    mode: str = "all_pairs",
    temporal_exclusion: int = 0,
) -> torch.Tensor:
    """Wang & Isola uniformity loss on unit-normalized embeddings.

    emb: (B, T, D) — unit vectors on S^{d-1}.
    Computes log mean exp(-t * ||mu_i - mu_j||^2) over a configurable subset
    of flattened batch-time pairs.
    """
    z = emb.reshape(-1, emb.size(-1))  # (B*T, D)
    if z.size(0) < 2:
        return z.new_tensor(0.0)

    sq_dists = torch.cdist(z, z, p=2).square()
    mask = _uniformity_pair_mask(
        emb.size(0),
        emb.size(1),
        mode=mode.lower(),
        temporal_exclusion=temporal_exclusion,
        device=z.device,
    )
    sq_dists = sq_dists[mask]
    if sq_dists.numel() == 0:
        return z.new_tensor(0.0)
    return torch.exp(-t * sq_dists).mean().log()


def infonce_loss(
    pred: torch.Tensor, target: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Symmetric batch-wise InfoNCE for next-step prediction.

    pred: (B, T, D) predicted next-step embeddings
    target: (B, T, D) ground-truth next-step embeddings

    For each time step t and batch item b, this uses the aligned pair
    (pred[b, t], target[b, t]) as the positive. The denominator contains:
    - all cross-view samples at the same time step, and
    - all same-view samples except self.

    The loss is symmetric:
      InfoNCE(pred, target) + InfoNCE(target, pred)
    averaged over batch and time.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} vs {target.shape}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    batch_size = pred.size(0)
    if batch_size < 2:
        return pred.new_tensor(0.0)

    q = F.normalize(pred, dim=-1, eps=1e-8)
    k = F.normalize(target, dim=-1, eps=1e-8)

    labels = torch.arange(batch_size, device=pred.device)
    labels = labels.unsqueeze(1).expand(-1, pred.size(1)).reshape(-1)
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=pred.device).unsqueeze(1)
    neg_inf = torch.tensor(-float("inf"), dtype=pred.dtype, device=pred.device)

    def directional_loss(query: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        cross_logits = torch.einsum("btd,jtd->btj", query, other) / temperature
        same_logits = torch.einsum("btd,jtd->btj", query, query) / temperature
        same_logits = same_logits.masked_fill(diag_mask, neg_inf)
        logits = torch.cat([cross_logits, same_logits], dim=-1)  # (B, T, 2B)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels)

    return 0.5 * (directional_loss(q, k) + directional_loss(k, q))


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
