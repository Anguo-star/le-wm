"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        pred_logvar_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # Optional scalar log-variance head for sigma-conditioned (heteroscedastic)
        # prediction. When None, predict_with_logvar() returns logvar=None and the
        # model behaves identically to the pure-MSE LeWM baseline.
        self.pred_logvar_proj = pred_logvar_proj

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def predict_with_logvar(self, emb, act_emb, *, detach_logvar_input: bool = False):
        """Predict next state embedding and (optional) per-token scalar log-variance.

        Both heads share the predictor backbone hidden state. Returns
        (mu_hat, logvar_hat) with shapes (B, T, D) and (B, T, 1) respectively.
        When pred_logvar_proj is None (default LeWM baseline), logvar_hat is
        None and callers must fall back to plain MSE prediction. In probe-only
        mode, detach_logvar_input prevents the sigma loss from updating the
        shared predictor backbone.
        """
        b = emb.size(0)
        hidden = self.predictor(emb, act_emb)              # (B, T, hidden)
        hidden_flat = rearrange(hidden, "b t d -> (b t) d")
        mu_hat = rearrange(self.pred_proj(hidden_flat), "(b t) d -> b t d", b=b)
        if self.pred_logvar_proj is None:
            return mu_hat, None
        logvar_input = hidden_flat.detach() if detach_logvar_input else hidden_flat
        logvar_flat = self.pred_logvar_proj(logvar_input)  # (B*T, 1)
        logvar_hat = rearrange(logvar_flat, "(b t) d -> b t d", b=b)
        return mu_hat, logvar_hat

    ####################
    ## Inference only ##
    ####################

    @staticmethod
    def _broadcast_goal_to_prediction(goal_emb: torch.Tensor, pred_emb: torch.Tensor) -> torch.Tensor:
        """Broadcast goal embeddings to match prediction rollout shape.

        `encode(goal)` typically returns `(B, T, D)`, while rollout predictions
        are shaped `(B, S, T, D)`. Planning cost only needs the last goal step,
        but we align the full tensor shape first so both JEPA and SphericalJEPA
        can share the same logic.
        """
        if goal_emb.ndim == pred_emb.ndim - 1:
            goal_emb = goal_emb.unsqueeze(1)
        if goal_emb.ndim != pred_emb.ndim:
            raise ValueError(
                f"goal_emb ndim {goal_emb.ndim} is incompatible with pred_emb ndim {pred_emb.ndim}"
            )
        return goal_emb[..., -1:, :].expand_as(pred_emb)

    def _inference_history_size(
        self, observed_history: int, history_size: int | None = None
    ) -> int:
        if history_size is not None:
            return int(history_size)
        for attr in ("history_size", "history_len"):
            value = getattr(self, attr, None)
            if value is not None:
                return int(value)
        pos_embedding = getattr(getattr(self, "predictor", None), "pos_embedding", None)
        if torch.is_tensor(pos_embedding) and pos_embedding.ndim >= 2:
            return int(pos_embedding.shape[1])
        return int(observed_history)

    def rollout(self, info, action_sequence, history_size: int | None = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = self._inference_history_size(H, history_size)
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = self._broadcast_goal_to_prediction(goal_emb, pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost


class SphericalJEPA(JEPA):
    """JEPA with L2-normalised (spherical) representations.

    Identical to JEPA except:
    - encode() exposes both pre-norm embeddings (`emb_raw`) and their
      L2-normalised version on S^{d-1} (`emb`).
    - predict_raw() returns pre-norm predictor outputs after `pred_proj`.
    - predict() returns the L2-normalised version of `predict_raw()`.
    - inference can use configurable rollout and cost spaces without changing
      the default normalized / cosine branch.

    Training still controls prediction and regularizer spaces explicitly in
    train_swm.py. The options below only affect planning / evaluation.
    """

    def __init__(
        self,
        *,
        inference_rollout_state_space: str = "normalized",
        inference_cost_space: str = "normalized",
        inference_cost_type: str = "cosine",
        analysis_prediction_space: str = "normalized",
        training_context_space: str = "normalized",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inference_rollout_state_space = inference_rollout_state_space.lower()
        self.inference_cost_space = inference_cost_space.lower()
        self.inference_cost_type = inference_cost_type.lower()
        self.analysis_prediction_space = analysis_prediction_space.lower()
        self.training_context_space = training_context_space.lower()

        valid_spaces = {"raw", "normalized", "sphere"}
        if self.inference_rollout_state_space not in valid_spaces:
            raise ValueError(
                "Unsupported inference_rollout_state_space: "
                f"{self.inference_rollout_state_space}"
            )
        if self.inference_cost_space not in valid_spaces:
            raise ValueError(f"Unsupported inference_cost_space: {self.inference_cost_space}")
        if self.inference_cost_type not in {"cosine", "mse"}:
            raise ValueError(f"Unsupported inference_cost_type: {self.inference_cost_type}")
        if self.analysis_prediction_space not in valid_spaces:
            raise ValueError(
                f"Unsupported analysis_prediction_space: {self.analysis_prediction_space}"
            )
        if self.training_context_space not in valid_spaces:
            raise ValueError(f"Unsupported training_context_space: {self.training_context_space}")

    def normalize_embeddings(self, emb):
        return F.normalize(emb, dim=-1, eps=1e-8)

    def _get_rollout_state_space(self) -> str:
        return getattr(self, "inference_rollout_state_space", "normalized").lower()

    def _get_cost_space(self) -> str:
        return getattr(self, "inference_cost_space", "normalized").lower()

    def _get_cost_type(self) -> str:
        return getattr(self, "inference_cost_type", "cosine").lower()

    def _resolve_space_name(self, space: str) -> str:
        if space == "sphere":
            return "normalized"
        return space

    def _select_embedding_space(self, *, emb_raw, emb_norm, space: str):
        space = self._resolve_space_name(space.lower())
        if space == "raw":
            return emb_raw
        if space == "normalized":
            return emb_norm
        raise ValueError(f"Unsupported embedding space: {space}")

    def encode(self, info):
        info = super().encode(info)
        info["emb_raw"] = info["emb"]
        info["emb"] = self.normalize_embeddings(info["emb_raw"])
        return info

    def predict_raw(self, emb, act_emb):
        return super().predict(emb, act_emb)

    def predict(self, emb, act_emb):
        preds = self.predict_raw(emb, act_emb)
        return self.normalize_embeddings(preds)

    def rollout(self, info, action_sequence, history_size: int | None = None):
        """Rollout while keeping raw and normalized trajectories side by side.

        `inference_rollout_state_space` decides what the predictor consumes
        autoregressively:
        - `normalized`: original spherical branch
        - `raw`: feed back raw predictor outputs directly
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        init_raw = _init["emb_raw"].unsqueeze(1).expand(B, S, -1, -1)
        init_norm = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

        rollout_raw = rearrange(init_raw, "b s ... -> (b s) ...").clone()
        rollout_norm = rearrange(init_norm, "b s ... -> (b s) ...").clone()
        rollout_state = self._select_embedding_space(
            emb_raw=rollout_raw,
            emb_norm=rollout_norm,
            space=self._get_rollout_state_space(),
        ).clone()

        act_full = rearrange(action_sequence, "b s ... -> (b s) ...")
        act_emb_full = self.action_encoder(act_full)

        HS = self._inference_history_size(H, history_size)
        for step in range(n_steps + 1):
            action_end = H + step
            # Match the original "take up to the last HS steps" behavior even
            # when the available prefix is shorter than HS.
            window = min(HS, action_end)
            emb_trunc = rollout_state[:, -window:]
            act_trunc = act_emb_full[:, action_end - window : action_end]
            pred_raw = self.predict_raw(emb_trunc, act_trunc)[:, -1:]
            pred_norm = self.normalize_embeddings(pred_raw)
            pred_state = self._select_embedding_space(
                emb_raw=pred_raw,
                emb_norm=pred_norm,
                space=self._get_rollout_state_space(),
            )

            rollout_raw = torch.cat([rollout_raw, pred_raw], dim=1)
            rollout_norm = torch.cat([rollout_norm, pred_norm], dim=1)
            rollout_state = torch.cat([rollout_state, pred_state], dim=1)

        info["predicted_emb_raw"] = rearrange(rollout_raw, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = rearrange(rollout_norm, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_rollout_state"] = rearrange(rollout_state, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict["goal_emb_raw"] = goal["emb_raw"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        return cost

    def criterion(self, info_dict: dict):
        """Configurable planning cost in raw or normalized space."""

        pred_emb = self._select_embedding_space(
            emb_raw=info_dict["predicted_emb_raw"],
            emb_norm=info_dict["predicted_emb"],
            space=self._get_cost_space(),
        )
        goal_emb = self._select_embedding_space(
            emb_raw=info_dict["goal_emb_raw"],
            emb_norm=info_dict["goal_emb"],
            space=self._get_cost_space(),
        )
        goal_emb = self._broadcast_goal_to_prediction(goal_emb, pred_emb)

        pred_last = pred_emb[..., -1:, :]
        goal_last = goal_emb[..., -1:, :].detach()

        if self._get_cost_type() == "cosine":
            cost = 1.0 - F.cosine_similarity(pred_last, goal_last, dim=-1)
            return cost.squeeze(-1)

        if self._get_cost_type() == "mse":
            cost = F.mse_loss(pred_last, goal_last, reduction="none").sum(
                dim=tuple(range(2, pred_last.ndim))
            )
            return cost

        raise ValueError(f"Unsupported inference_cost_type: {self._get_cost_type()}")
