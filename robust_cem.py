"""Planner-side robust CEM variants for evaluation-time interventions.

The default path is intentionally identical to stable-worldmodel's CEMSolver.
Risk-aware behavior is enabled only through ``robust.enabled=true``.
"""

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging


@dataclass
class RobustCEMConfig:
    enabled: bool = False
    mode: str = "rerank_topk"
    topk: int = 30
    robust_current: bool = True
    robust_goal: bool = True
    belief_mode: str = "input_tta_empirical"
    tta_num: int = 8
    tta_noise_std: float = 0.005
    tta_include_identity: bool = True
    latent_samples: int = 8
    risk: str = "cvar"
    cvar_q: float = 0.8
    quantile_q: float = 0.8
    lambda_std: float = 1.0
    log_debug: bool = True
    robust_history_limit: int = 256
    tta_clamp: bool = True


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _robust_config(cfg: Any) -> RobustCEMConfig:
    return RobustCEMConfig(
        enabled=bool(_cfg_get(cfg, "enabled", False)),
        mode=str(_cfg_get(cfg, "mode", "rerank_topk")),
        topk=int(_cfg_get(cfg, "topk", 30)),
        robust_current=bool(_cfg_get(cfg, "robust_current", True)),
        robust_goal=bool(_cfg_get(cfg, "robust_goal", True)),
        belief_mode=str(_cfg_get(cfg, "belief_mode", "input_tta_empirical")),
        tta_num=int(_cfg_get(cfg, "tta_num", 8)),
        tta_noise_std=float(_cfg_get(cfg, "tta_noise_std", 0.005)),
        tta_include_identity=bool(_cfg_get(cfg, "tta_include_identity", True)),
        latent_samples=int(_cfg_get(cfg, "latent_samples", 8)),
        risk=str(_cfg_get(cfg, "risk", "cvar")),
        cvar_q=float(_cfg_get(cfg, "cvar_q", 0.8)),
        quantile_q=float(_cfg_get(cfg, "quantile_q", 0.8)),
        lambda_std=float(_cfg_get(cfg, "lambda_std", 1.0)),
        log_debug=bool(_cfg_get(cfg, "log_debug", True)),
        robust_history_limit=int(_cfg_get(cfg, "robust_history_limit", 256)),
        tta_clamp=bool(_cfg_get(cfg, "tta_clamp", True)),
    )


def aggregate_risk(
    cost_samples: torch.Tensor,
    risk: str,
    *,
    lambda_std: float = 1.0,
    q: float = 0.8,
) -> torch.Tensor:
    """Aggregate per-candidate cost samples into a scalar risk.

    Args:
        cost_samples: Tensor with shape ``(..., S)`` where S is the number of
            belief/TTA samples.
        risk: One of ``mean``, ``mean_std``, ``quantile``, ``cvar``, ``max``.
    """
    if cost_samples.ndim < 1:
        raise ValueError("cost_samples must have at least one dimension")

    risk = risk.lower()
    if risk == "mean":
        return cost_samples.mean(dim=-1)
    if risk == "mean_std":
        return cost_samples.mean(dim=-1) + lambda_std * cost_samples.std(dim=-1, unbiased=False)
    if risk == "quantile":
        return torch.quantile(cost_samples, q, dim=-1)
    if risk == "cvar":
        threshold = torch.quantile(cost_samples, q, dim=-1, keepdim=True)
        mask = cost_samples >= threshold
        return (cost_samples * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    if risk == "max":
        return cost_samples.max(dim=-1).values
    raise ValueError(f"Unsupported robust CEM risk: {risk}")


class RiskAwareCEMSolver:
    """CEM with optional final-elite risk reranking under input TTA.

    ``robust.enabled=false`` preserves stable-worldmodel CEM behavior. When
    enabled, the normal CEM optimization is left untouched; only the final elite
    candidates are re-evaluated under small perturbations of the already-observed
    transformed pixels/goal tensors.
    """

    def __init__(
        self,
        model: Any,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        callbacks: list[Any] | None = None,
        robust: Any | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = torch.device(device)
        self._dtype = self._infer_model_dtype()
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.cpu_gen = torch.Generator(device="cpu").manual_seed(seed + 100003)
        self.callbacks = list(callbacks) if callbacks else []
        self.robust = _robust_config(robust)
        self.last_robust_stats: dict[str, Any] = {}
        self.robust_history: list[dict[str, Any]] = []

        if self.robust.enabled and self.robust.mode != "rerank_topk":
            raise ValueError("RiskAwareCEMSolver currently supports only mode=rerank_topk")
        if self.robust.enabled and self.robust.belief_mode not in {"input_tta_empirical", "none"}:
            raise ValueError(
                "Stage-1 robust CEM supports belief_mode=input_tta_empirical or none. "
                f"Got {self.robust.belief_mode!r}."
            )

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(
                f"Action space is discrete, got {type(action_space)}. "
                "RiskAwareCEMSolver may not work as expected."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _infer_model_dtype(self) -> torch.dtype:
        if hasattr(self.model, "parameters"):
            try:
                return next(self.model.parameters()).dtype
            except StopIteration:
                pass
        return torch.float32

    def _tensor_to_solver(self, value: torch.Tensor) -> torch.Tensor:
        target_dtype = self.dtype if value.is_floating_point() else None
        return value.to(device=self.device, dtype=target_dtype)

    def prepare_init_action(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        *,
        n_envs: int,
    ) -> torch.Tensor | None:
        """Normalize warm-start actions and fill missing horizon tail.

        This mirrors stable-worldmodel's prepare_init_action: Costable-only
        checkpoints zero-pad, while Costable+Actionable models can fill the
        missing tail through ``get_action(..., horizon=, prefix_actions=)``.
        """
        if init_action is None:
            actions = None
        else:
            actions = init_action.to(device=self.device, dtype=self.dtype)
            assert actions.shape[0] == n_envs, (
                f"init_action batch size {actions.shape[0]} != n_envs {n_envs}"
            )
            assert actions.shape[2] == self.action_dim, (
                f"init_action action_dim {actions.shape[2]} != action_dim {self.action_dim}"
            )

        n_prev = actions.shape[1] if actions is not None else 0
        remaining = self.horizon - n_prev
        if remaining <= 0:
            return actions[:, : self.horizon]

        tail = self._actionable_warm_start_tail(info_dict, remaining, actions, n_envs=n_envs)
        if tail is None:
            tail = torch.zeros(
                [n_envs, remaining, self.action_dim],
                device=self.device,
                dtype=self.dtype,
            )
        if actions is not None:
            return torch.cat([actions.to(tail.device), tail], dim=1)
        return tail

    def _actionable_warm_start_tail(
        self,
        info_dict: dict,
        remaining: int,
        prefix_actions: torch.Tensor | None,
        *,
        n_envs: int,
    ) -> torch.Tensor | None:
        if not hasattr(self.model, "get_action"):
            return None

        prepared_info = {}
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                prepared_info[k] = self._tensor_to_solver(v)
            else:
                prepared_info[k] = v

        try:
            tail = self.model.get_action(
                prepared_info,
                horizon=remaining,
                prefix_actions=prefix_actions,
            )
        except TypeError as exc:  # pragma: no cover - legacy Actionable fallback
            logging.warning(
                "Actionable warm-start did not accept horizon/prefix_actions; "
                f"falling back to repeated one-step get_action: {exc}"
            )
            return self._legacy_actionable_tail(prepared_info, remaining, n_envs=n_envs)
        except Exception as exc:  # pragma: no cover - compatibility fallback
            logging.warning(f"Actionable warm-start failed; falling back to zeros: {exc}")
            return None

        if isinstance(tail, np.ndarray):
            tail = torch.from_numpy(tail)
        if not torch.is_tensor(tail):
            tail = torch.as_tensor(tail)
        tail = tail.to(device=self.device, dtype=self.dtype)
        if tail.ndim == 2:
            tail = tail.unsqueeze(1)
        if tail.shape != (n_envs, remaining, self.action_dim):
            logging.warning(
                "Actionable warm-start produced shape "
                f"{tuple(tail.shape)}, expected {(n_envs, remaining, self.action_dim)}; "
                "falling back to zeros."
            )
            return None
        return tail

    def _legacy_actionable_tail(
        self, prepared_info: dict, remaining: int, *, n_envs: int
    ) -> torch.Tensor | None:
        tail = []
        for _ in range(remaining):
            try:
                action = self.model.get_action(prepared_info)
            except Exception as exc:  # pragma: no cover - compatibility fallback
                logging.warning(f"Legacy Actionable warm-start failed; falling back to zeros: {exc}")
                return None
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            if not torch.is_tensor(action):
                action = torch.as_tensor(action)
            action = action.to(device=self.device, dtype=self.dtype).reshape(n_envs, -1)
            if action.shape[-1] == self._action_dim:
                action = action.repeat(1, self._config.action_block)
            if action.shape[-1] != self.action_dim:
                logging.warning(
                    "Legacy Actionable warm-start produced action dim "
                    f"{action.shape[-1]}, expected {self.action_dim}; falling back to zeros."
                )
                return None
            tail.append(action.unsqueeze(1))
        return torch.cat(tail, dim=1) if tail else None

    def init_action_distrib(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim],
            device=self.device,
            dtype=self.dtype,
        )
        mean = (
            torch.zeros(
                [n_envs, 0, self.action_dim],
                device=self.device,
                dtype=self.dtype,
            )
            if actions is None
            else actions.to(device=self.device, dtype=self.dtype)
        )

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            new_mean = torch.zeros(
                [n_envs, remaining, self.action_dim],
                device=self.device,
                dtype=self.dtype,
            )
            mean = torch.cat([mean, new_mean], dim=1)

        return mean, var

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        start_time = time.time()
        outputs = {
            "costs": [],
            "mean": [],
            "var": [],
        }

        total_envs = len(next(iter(info_dict.values())))
        init_action = self.prepare_init_action(info_dict, init_action, n_envs=total_envs)
        mean, var = self.init_action_distrib(total_envs, init_action)
        robust_stats: list[dict[str, Any]] = []

        for cb in self.callbacks:
            cb.reset()

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            expanded_infos = self._expand_info(info_dict, start_idx, end_idx, self.num_samples)

            final_batch_cost = None
            final_topk_candidates = None
            final_topk_point_costs = None

            for cb in self.callbacks:
                cb.start_batch()

            for _step in range(self.n_steps):
                is_last_step = _step == self.n_steps - 1
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                current_info = expanded_infos.copy()
                costs = self.model.get_cost(current_info, candidates)

                assert isinstance(costs, torch.Tensor), (
                    f"Expected cost to be a torch.Tensor, got {type(costs)}"
                )
                assert costs.ndim == 2 and costs.shape[0] == current_bs and costs.shape[1] == self.num_samples, (
                    f"Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}"
                )

                select_k = self.topk
                if self.robust.enabled and is_last_step:
                    select_k = min(self.num_samples, max(self.topk, int(self.robust.topk)))
                topk_vals, topk_inds = torch.topk(costs, k=select_k, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, select_k)
                topk_candidates = candidates[batch_indices, topk_inds]

                elite_candidates = topk_candidates[:, : self.topk]
                elite_vals = topk_vals[:, : self.topk]
                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = elite_candidates.mean(dim=1)
                batch_var = elite_candidates.std(dim=1)

                for cb in self.callbacks:
                    cb(
                        step=_step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=elite_vals,
                        topk_inds=topk_inds[:, : self.topk],
                        topk_candidates=elite_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                    )

                final_batch_cost = elite_vals.mean(dim=1).cpu().tolist()
                final_topk_candidates = topk_candidates
                final_topk_point_costs = topk_vals

            if self.robust.enabled:
                assert final_topk_candidates is not None
                reranked, stats = self._rerank_topk(
                    info_dict=info_dict,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    topk_candidates=final_topk_candidates,
                    point_costs=final_topk_point_costs,
                )
                batch_mean = reranked
                robust_stats.extend(stats)

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        if self.callbacks:
            outputs["callbacks"] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs["callbacks"][cb.output_key] = cb.history

        self.last_robust_stats = {
            "enabled": self.robust.enabled,
            "mode": self.robust.mode,
            "belief_mode": self.robust.belief_mode,
            "risk": self.robust.risk,
            "samples": self._num_risk_samples(),
            "batches": robust_stats,
        }
        if self.robust.enabled:
            self.robust_history.append(self.last_robust_stats)
            if self.robust.robust_history_limit > 0:
                self.robust_history = self.robust_history[-self.robust.robust_history_limit :]
            outputs["robust"] = self.last_robust_stats

        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs

    def _expand_info(self, info_dict: dict, start_idx: int, end_idx: int, num_samples: int) -> dict:
        current_bs = end_idx - start_idx
        expanded_infos = {}
        for k, v in info_dict.items():
            v_batch = v[start_idx:end_idx]
            if torch.is_tensor(v):
                v_batch = self._tensor_to_solver(v_batch).unsqueeze(1)
                v_batch = v_batch.expand(current_bs, num_samples, *v_batch.shape[2:])
            elif isinstance(v, np.ndarray):
                v_batch = np.repeat(v_batch[:, None, ...], num_samples, axis=1)
            expanded_infos[k] = v_batch
        return expanded_infos

    def _num_risk_samples(self) -> int:
        if self.robust.belief_mode == "none":
            return 1
        if self.robust.belief_mode == "input_tta_empirical":
            return max(1, self.robust.tta_num)
        return max(1, self.robust.latent_samples)

    def _rerank_topk(
        self,
        *,
        info_dict: dict,
        start_idx: int,
        end_idx: int,
        topk_candidates: torch.Tensor,
        point_costs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        rerank_k = min(int(self.robust.topk), topk_candidates.shape[1])
        candidates = topk_candidates[:, :rerank_k]
        base_info = self._expand_info(info_dict, start_idx, end_idx, rerank_k)

        cost_samples = []
        for sample_idx in range(self._num_risk_samples()):
            robust_info = self._make_risk_sample_info(base_info, sample_idx)
            costs = self.model.get_cost(robust_info, candidates)
            cost_samples.append(costs)

        stacked = torch.stack(cost_samples, dim=-1)
        q = self.robust.cvar_q if self.robust.risk == "cvar" else self.robust.quantile_q
        risk_cost = aggregate_risk(
            stacked,
            self.robust.risk,
            lambda_std=self.robust.lambda_std,
            q=q,
        )
        best_idx = risk_cost.argmin(dim=1)
        batch_idx = torch.arange(candidates.shape[0], device=candidates.device)
        selected = candidates[batch_idx, best_idx]

        stats = []
        for b in range(candidates.shape[0]):
            selected_point_cost = None
            if point_costs is not None:
                selected_point_cost = float(point_costs[b, best_idx[b]].detach().cpu())
            selected_risk = float(risk_cost[b, best_idx[b]].detach().cpu())
            stats.append(
                {
                    "selected_candidate_idx": int(best_idx[b].detach().cpu()),
                    "risk_cost_mean": float(risk_cost[b].mean().detach().cpu()),
                    "risk_cost_std": float(risk_cost[b].std(unbiased=False).detach().cpu()),
                    "selected_candidate_point_cost": selected_point_cost,
                    "selected_candidate_risk_cost": selected_risk,
                    "risk_minus_point_cost": (
                        None if selected_point_cost is None else selected_risk - selected_point_cost
                    ),
                }
            )
        return selected, stats

    def _make_risk_sample_info(self, base_info: dict, sample_idx: int) -> dict:
        if self.robust.belief_mode == "none":
            return base_info.copy()

        info = {}
        for k, v in base_info.items():
            if torch.is_tensor(v):
                v = v.clone()
                if k == "pixels" and self.robust.robust_current:
                    v = self._add_tta_noise(v, sample_idx)
                elif k == "goal" and self.robust.robust_goal:
                    v = self._add_tta_noise(v, sample_idx)
            info[k] = v
        return info

    def _add_tta_noise(self, value: torch.Tensor, sample_idx: int) -> torch.Tensor:
        if self.robust.tta_include_identity and sample_idx == 0:
            return value
        if self.robust.tta_noise_std <= 0:
            return value

        noise_shape = list(value.shape)
        if len(noise_shape) >= 2:
            noise_shape[1] = 1
        generator = self.cpu_gen if value.device.type == "cpu" else self.torch_gen
        noise = torch.randn(
            noise_shape,
            generator=generator,
            device=value.device,
            dtype=value.dtype,
        )
        if len(noise_shape) >= 2 and value.shape[1] != 1:
            noise = noise.expand_as(value)
        noised = value + noise * self.robust.tta_noise_std
        if not self.robust.tta_clamp:
            return noised

        # Inputs are already ImageNet-normalized in eval.py, so [0, 1] would be
        # the wrong range here. Clamp to the observed transformed tensor range
        # per image/token to prevent TTA samples from leaving the local support.
        feature_start = 3 if value.ndim >= 6 else 2
        reduce_dims = tuple(range(feature_start, value.ndim))
        if not reduce_dims:
            return noised
        lower = value.amin(dim=reduce_dims, keepdim=True)
        upper = value.amax(dim=reduce_dims, keepdim=True)
        return torch.minimum(torch.maximum(noised, lower), upper)
