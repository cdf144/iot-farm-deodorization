#!/usr/bin/env python3
"""
IoT Farm Deodorization RL Training Script with Subprocess Parallelization

Focuses on RL tuning, training, and evaluation stages.
Uses use_subproc=True for parallel environment execution.
Logs all outputs to training_log.txt and results.csv
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import optuna
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# ==============================================================================
# SETUP & LOGGING
# ==============================================================================

BASE_DIR = Path.cwd()
LOG_FILE = BASE_DIR / "training_log.txt"
RESULTS_FILE = BASE_DIR / "results.csv"

# Setup dual logging: file + console
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_FILE)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def log_info(msg: str):
    """Centralized logging."""
    logger.info(msg)


# ==============================================================================
# CONSTANTS & CONFIGS
# ==============================================================================

ACTION_CONCENTRATIONS: tuple[int, ...] = (0, 500, 1000, 1500)

TVOC_EXCESS_NORM = 17.5
CO2_EXCESS_NORM = 144.0
THI_EXCESS_NORM = 0.045
RH_EXCESS_NORM = 5.75
COLD_EXCESS_NORM = 0.040
TVOC_PROGRESS_NORM = 4.75
CO2_PROGRESS_NORM = 16.2

AlgoName = Literal["ppo", "dqn"]


@dataclass(frozen=True)
class RewardWeights:
    tvoc_excess_scale: float = 1.2
    tvoc_progress_scale: float = 1.0
    use_tvoc_delta: bool = True
    tvoc_target: float = 50.0

    co2_excess_scale: float = 0.6
    co2_progress_scale: float = 0.4
    use_co2_delta: bool = True
    co2_target: float = 400.0

    thi_scale: float = 0.45
    rh_excess_scale: float = 0.15
    rh_target: float = 90.0
    cold_scale: float = 0.30

    action_scale: float = 0.9
    action_curve_power: float = 1.5
    action_urgency_scale: float = 1.5


@dataclass
class Pi1SimulatorConfig:
    gamma: float = 0.99
    baseline_decay_frac_per_step: float = 0.40
    co2_baseline_decay_frac_per_step: float = 0.40
    time_step_minutes: float = 5.0
    spray_volume_ml: float = 1000.0
    alpha_tvoc: float = 0.0005
    alpha_rh: float = 0.0005
    alpha_temp: float = 0.0005
    alpha_co2: float = 0.00002
    tvoc_noise_sigma: float = 2.35
    rh_noise_sigma: float = 0.10
    co2_noise_sigma: float = 4.92
    temp_noise_sigma: float = 0.023
    reward: RewardWeights = field(default_factory=RewardWeights)


@dataclass(frozen=True)
class RLRunConfig:
    max_steps: int = 250
    tune_trials: int = 20
    tune_budget: int = 50_000
    eval_episodes: int = 20
    budgets: tuple[int, ...] = (50_000, 100_000, 200_000)
    n_envs: int = 4
    seed: int = 42
    use_subproc: bool = True  # KEY: use subprocess for parallel envs


@dataclass(frozen=True)
class EnvConfig:
    reward: RewardWeights = field(default_factory=RewardWeights)
    max_steps: int = 250
    gamma: float = 0.99
    tvoc_norm_divisor: float = 1.0
    co2_norm_divisor: float = 1.0
    temp_norm_divisor: float = 1.0
    rh_min: float = 0.0
    rh_span: float = 1.0


# ==============================================================================
# SIMULATOR & ENVIRONMENT
# ==============================================================================


def wetbulb_stull(temp_c: float, rh: float) -> float:
    """Compute wet bulb temperature using Stull 2011 approximation."""
    return (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


class Pi1SpraySimulator:
    """Mechanistic simulator for RL prototyping on pi-1 5-minute data."""

    def __init__(
        self,
        pi1_df: pd.DataFrame,
        config: Pi1SimulatorConfig | None = None,
        seed: int = 7,
    ) -> None:
        self.cfg = config or Pi1SimulatorConfig()
        self.rng = np.random.default_rng(seed)

        required_cols = ["tvoc", "co2", "temp", "humidity"]
        base = pi1_df.copy()
        for col in required_cols:
            base[col] = pd.to_numeric(base[col], errors="coerce")
        base = base.dropna(subset=required_cols).reset_index(drop=True)

        self.real_tvoc = base["tvoc"].to_numpy(dtype=float)
        self.real_co2 = base["co2"].to_numpy(dtype=float)
        self.real_temp = base["temp"].to_numpy(dtype=float)
        self.real_rh = base["humidity"].to_numpy(dtype=float)
        self.T = len(base)

        # Inverse Box Model: Recover latent emissions from real data
        self.emission = np.empty(self.T, dtype=float)
        self.emission[:-1] = (
            self.real_tvoc[1:]
            - self.real_tvoc[:-1]
            + self.cfg.baseline_decay_frac_per_step * self.real_tvoc[:-1]
        )
        self.emission[-1] = self.emission[-2]

        self.co2_emission = np.empty(self.T, dtype=float)
        self.co2_emission[:-1] = (
            self.real_co2[1:]
            - self.real_co2[:-1]
            + self.cfg.co2_baseline_decay_frac_per_step * self.real_co2[:-1]
        )
        self.co2_emission[-1] = self.co2_emission[-2]

    @staticmethod
    def action_space() -> list[int]:
        return list(ACTION_CONCENTRATIONS)

    def reset(self, t0: int = 0) -> np.ndarray:
        t0 = int(np.clip(t0, 0, self.T - 1))
        return np.array(
            [
                self.real_tvoc[t0],
                self.real_co2[t0],
                self.real_temp[t0],
                self.real_rh[t0],
            ],
            dtype=float,
        )

    def step(self, state: np.ndarray, action: int, t: int) -> np.ndarray:
        t = int(np.clip(t, 0, self.T - 2))
        tvoc_current = float(state[0])
        co2_current = float(state[1])

        spray_volume_L = self.cfg.spray_volume_ml / 1000.0
        dose_mg = float(action) * spray_volume_L

        k_base = (
            -np.log(1.0 - self.cfg.baseline_decay_frac_per_step)
            / self.cfg.time_step_minutes
        )
        tvoc_spray_loss = self.cfg.alpha_tvoc * dose_mg * tvoc_current
        tvoc_next = (
            tvoc_current * np.exp(-k_base * self.cfg.time_step_minutes)
            + self.emission[t]
            - tvoc_spray_loss
        )
        tvoc_next = max(
            0.0, tvoc_next + self.rng.normal(0.0, self.cfg.tvoc_noise_sigma)
        )

        k_co2_base = (
            -np.log(1.0 - self.cfg.co2_baseline_decay_frac_per_step)
            / self.cfg.time_step_minutes
        )
        k_co2_spray = self.cfg.alpha_co2 * dose_mg
        k_co2_total = k_co2_base + k_co2_spray
        co2_next = (
            co2_current * np.exp(-k_co2_total * self.cfg.time_step_minutes)
            + self.co2_emission[t]
        )
        co2_next = max(300.0, co2_next + self.rng.normal(0.0, self.cfg.co2_noise_sigma))

        rh_base = self.real_rh[t + 1]
        temp_base = self.real_temp[t + 1]

        rh_next = rh_base + self.cfg.alpha_rh * dose_mg * (100.0 - rh_base)
        rh_next = rh_next + self.rng.normal(0.0, self.cfg.rh_noise_sigma)
        rh_next = float(np.clip(rh_next, 0.0, 100.0))

        wetbulb = wetbulb_stull(temp_base, rh_base)
        temp_drop = self.cfg.alpha_temp * dose_mg * (temp_base - wetbulb)
        temp_next = max(
            wetbulb,
            temp_base - temp_drop + self.rng.normal(0.0, self.cfg.temp_noise_sigma),
        )

        return np.array([tvoc_next, co2_next, temp_next, rh_next], dtype=float)

    def compute_reward(
        self,
        tvoc_prev: float,
        tvoc_next: float,
        co2_prev: float,
        co2_next: float,
        temp_prev: float,
        temp_next: float,
        rh_prev: float,
        rh_next: float,
        action: int,
        reward_weights: RewardWeights | None = None,
        is_terminal: bool = False,
    ) -> float:
        w = reward_weights or self.cfg.reward

        tvoc_excess = max(0.0, tvoc_next - w.tvoc_target) / TVOC_EXCESS_NORM
        tvoc_excess_term = -w.tvoc_excess_scale * tvoc_excess

        co2_excess = max(0.0, co2_next - w.co2_target) / CO2_EXCESS_NORM
        co2_excess_term = -w.co2_excess_scale * co2_excess

        thi_prev = (0.8 * temp_prev) + ((rh_prev / 100.0) * (temp_prev - 14.4)) + 46.4
        thi_next = (0.8 * temp_next) + ((rh_next / 100.0) * (temp_next - 14.4)) + 46.4

        thi_term = (
            -w.thi_scale * max(0.0, thi_next - max(73.0, thi_prev)) / THI_EXCESS_NORM
        )
        rh_term = -w.rh_excess_scale * max(0.0, rh_next - w.rh_target) / RH_EXCESS_NORM
        cold_term = -w.cold_scale * max(0.0, 18.0 - temp_next) / COLD_EXCESS_NORM

        tvoc_urgency = np.clip(
            (tvoc_prev - w.tvoc_target) / max(1.0, w.tvoc_target), 0.0, 1.0
        )
        dose_norm = float(action) / 1500.0
        urgency_multiplier = 1.0 + w.action_urgency_scale * (1.0 - tvoc_urgency)
        action_term = (
            -w.action_scale * (dose_norm**w.action_curve_power) * urgency_multiplier
        )

        base_reward = (
            tvoc_excess_term
            + co2_excess_term
            + thi_term
            + rh_term
            + cold_term
            + action_term
        )

        phi_prev = self._phi(tvoc_prev, co2_prev, temp_prev, rh_prev, w)
        phi_next = self._phi(tvoc_next, co2_next, temp_next, rh_next, w)
        shaping = 0.0 if is_terminal else (self.cfg.gamma * phi_next - phi_prev)

        return float(base_reward + shaping)

    def _phi(
        self, tvoc: float, co2: float, temp: float, rh: float, w: RewardWeights
    ) -> float:
        thi = (0.8 * temp) + ((rh / 100.0) * (temp - 14.4)) + 46.4

        tvoc_excess = max(0.0, tvoc - w.tvoc_target) / TVOC_EXCESS_NORM
        co2_excess = max(0.0, co2 - w.co2_target) / CO2_EXCESS_NORM
        thi_excess = max(0.0, thi - 73.0) / THI_EXCESS_NORM
        rh_excess = max(0.0, rh - w.rh_target) / RH_EXCESS_NORM
        cold_excess = max(0.0, 18.0 - temp) / COLD_EXCESS_NORM

        return -(
            w.tvoc_excess_scale * tvoc_excess
            + w.co2_excess_scale * co2_excess
            + w.thi_scale * thi_excess
            + w.rh_excess_scale * rh_excess
            + w.cold_scale * cold_excess
        )


class Pi1SprayEnv(gym.Env):
    """Gymnasium-compliant environment for deodorization spray control."""

    metadata = {"render_modes": []}

    def __init__(
        self, simulator: Pi1SpraySimulator, config: EnvConfig | None = None
    ) -> None:
        self.sim = simulator
        self.cfg = config or EnvConfig()

        self.action_to_concentration = {
            idx: concentration
            for idx, concentration in enumerate(ACTION_CONCENTRATIONS)
        }
        self.action_space = spaces.Discrete(len(self.action_to_concentration))

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.state: np.ndarray = np.zeros(4, dtype=np.float32)
        self.t: int = 0
        self.steps_taken: int = 0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        _ = options
        start_t = self.np_random.integers(0, max(1, self.sim.T - self.cfg.max_steps))
        self.t = start_t
        self.state = self.sim.reset(t0=start_t).astype(np.float32)
        self.steps_taken = 0

        obs = self._build_obs(self.state, d_tvoc_norm=0.0, d_rh_norm=0.0)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        dose = self.action_to_concentration[int(action)]
        state_prev = self.state.copy()
        next_state = self.sim.step(state_prev, dose, self.t).astype(np.float32)

        d_tvoc_norm = float(np.clip((next_state[0] - state_prev[0]) / 500.0, -1.0, 1.0))
        d_rh_norm = float(np.clip((next_state[3] - state_prev[3]) / 60.0, -1.0, 1.0))

        self.state = next_state
        self.t += 1
        self.steps_taken += 1

        terminated = self.t >= self.sim.T - 2
        truncated = self.steps_taken >= self.cfg.max_steps
        is_terminal = terminated or truncated

        reward = self.sim.compute_reward(
            tvoc_prev=float(state_prev[0]),
            tvoc_next=float(next_state[0]),
            co2_prev=float(state_prev[1]),
            co2_next=float(next_state[1]),
            temp_prev=float(state_prev[2]),
            temp_next=float(next_state[2]),
            rh_prev=float(state_prev[3]),
            rh_next=float(next_state[3]),
            action=dose,
            reward_weights=self.cfg.reward,
            is_terminal=is_terminal,
        )

        info = {
            "tvoc": float(self.state[0]),
            "co2": float(self.state[1]),
            "temp": float(self.state[2]),
            "rh": float(self.state[3]),
            "dose": dose,
            "d_tvoc_norm": d_tvoc_norm,
            "d_rh_norm": d_rh_norm,
        }

        obs = self._build_obs(self.state, d_tvoc_norm=d_tvoc_norm, d_rh_norm=d_rh_norm)
        return obs, reward, terminated, truncated, info

    def _build_obs(
        self, s: np.ndarray, d_tvoc_norm: float, d_rh_norm: float
    ) -> np.ndarray:
        base = np.array(
            [
                s[0] / self.cfg.tvoc_norm_divisor,
                s[1] / self.cfg.co2_norm_divisor,
                s[2] / self.cfg.temp_norm_divisor,
                (s[3] - self.cfg.rh_min) / self.cfg.rh_span,
            ],
            dtype=np.float32,
        )
        delta = np.array([d_tvoc_norm, d_rh_norm], dtype=np.float32)
        return np.concatenate([base, delta], axis=0)


# ==============================================================================
# RL INFRASTRUCTURE
# ==============================================================================

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN}


def make_env_cfg(sim: Pi1SpraySimulator, gamma: float, obs_divisors: dict) -> EnvConfig:
    return EnvConfig(
        reward=sim.cfg.reward,
        max_steps=RLRunConfig.max_steps,
        gamma=gamma,
        tvoc_norm_divisor=obs_divisors.get("tvoc", 1.0),
        co2_norm_divisor=obs_divisors.get("co2", 1.0),
        temp_norm_divisor=obs_divisors.get("temp", 1.0),
        rh_min=obs_divisors.get("rh_min", 0.0),
        rh_span=obs_divisors.get("rh_span", 1.0),
    )


def make_env_fn(base_sim, env_cfg: EnvConfig, seed: int, rank: int):
    def _init():
        sim_i = copy.deepcopy(base_sim)
        sim_i.rng = np.random.default_rng(seed + 10_000 * rank)
        env = Pi1SprayEnv(sim_i, config=env_cfg)
        return Monitor(env)

    return _init


def build_vec_env(
    base_sim,
    env_cfg: EnvConfig,
    seed: int,
    n_envs: int,
    normalize: bool = True,
    use_subproc: bool = False,
    gamma: float = 0.99,
):
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    env_fns = [make_env_fn(base_sim, env_cfg, seed, rank) for rank in range(n_envs)]
    vec_env = vec_cls(env_fns)

    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
        )
    return vec_env


def sync_vecnormalize(train_env, eval_env):
    eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)
    eval_env.ret_rms = copy.deepcopy(train_env.ret_rms)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def make_eval_env(base_sim, env_cfg: EnvConfig, train_env, seed: int):
    eval_env = build_vec_env(
        base_sim=base_sim,
        env_cfg=env_cfg,
        seed=seed,
        n_envs=1,
        normalize=True,
        use_subproc=False,
        gamma=env_cfg.gamma,
    )
    return sync_vecnormalize(train_env, eval_env)


def sample_hyperparams(algo: AlgoName, trial: optuna.Trial) -> dict:
    if algo == "ppo":
        net_arch_str = trial.suggest_categorical(
            "net_arch", ["64,64", "128,128", "256,256"]
        )
        net_arch = tuple(int(x) for x in net_arch_str.split(","))
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 10]),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.98),
            "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 2e-2, log=True),
            "policy_kwargs": {"net_arch": list(net_arch)},
        }

    if algo == "dqn":
        net_arch_str = trial.suggest_categorical(
            "net_arch", ["64,64", "128,128", "256,256"]
        )
        net_arch = tuple(int(x) for x in net_arch_str.split(","))
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_categorical(
                "buffer_size", [50_000, 100_000, 200_000]
            ),
            "learning_starts": trial.suggest_categorical(
                "learning_starts", [1_000, 5_000, 10_000]
            ),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4]),
            "target_update_interval": trial.suggest_categorical(
                "target_update_interval", [250, 500, 1000]
            ),
            "exploration_fraction": trial.suggest_float(
                "exploration_fraction", 0.05, 0.30
            ),
            "exploration_final_eps": trial.suggest_float(
                "exploration_final_eps", 0.01, 0.10
            ),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
            "policy_kwargs": {"net_arch": list(net_arch)},
        }

    raise ValueError(f"Unknown algo: {algo}")


def finalize_params(params: dict) -> dict:
    params = dict(params)
    if "net_arch" in params:
        raw = params.pop("net_arch")
        if isinstance(raw, str):
            net_arch = tuple(int(x) for x in raw.split(","))
        else:
            net_arch = tuple(raw)
        params["policy_kwargs"] = {"net_arch": list(net_arch)}
    return params


def build_model(algo: AlgoName, train_env, params: dict, seed: int):
    cls = ALGO_CLASSES[algo]
    common = dict(
        policy="MlpPolicy",
        env=train_env,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    return cls(**common, **params)


def evaluate_model(
    model,
    eval_env,
    env_cfg: EnvConfig,
    n_episodes: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    reward_list: list[float] = []
    tvoc_delta_list: list[float] = []
    rh_excess_list: list[float] = []
    co2_list: list[float] = []
    temp_list: list[float] = []
    action_counts = {dose: 0 for dose in ACTION_CONCENTRATIONS}
    step_tvoc_hit = 0
    step_co2_hit = 0
    step_count = 0

    for ep in range(n_episodes):
        eval_env.seed(seed + ep)
        obs = eval_env.reset()
        tvoc_start = float(eval_env.get_attr("state")[0][0])
        ep_reward = 0.0
        last_info = None
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = eval_env.step(action)

            ep_reward += float(rewards[0])
            info = infos[0]
            last_info = info

            dose = int(info["dose"])
            action_counts[dose] += 1

            tvoc = float(info["tvoc"])
            co2 = float(info["co2"])
            temp = float(info["temp"])
            rh = float(info["rh"])

            tvoc_delta_list.append(tvoc - tvoc_start)
            rh_excess_list.append(max(0.0, rh - env_cfg.reward.rh_target))
            co2_list.append(co2)
            temp_list.append(temp)

            if tvoc <= env_cfg.reward.tvoc_target:
                step_tvoc_hit += 1
            if co2 <= env_cfg.reward.co2_target:
                step_co2_hit += 1
            step_count += 1

        reward_list.append(ep_reward)

    total_actions = max(1, sum(action_counts.values()))
    action_ratio = {
        f"concentration_{dose}_ratio": action_counts[dose] / total_actions
        for dose in ACTION_CONCENTRATIONS
    }
    action_probs = np.array(
        [action_counts[d] / total_actions for d in ACTION_CONCENTRATIONS], dtype=float
    )
    action_entropy = float(
        -(action_probs[action_probs > 0] * np.log(action_probs[action_probs > 0])).sum()
    )

    out = {
        "episodes": n_episodes,
        "reward_mean": float(np.mean(reward_list)),
        "reward_std": float(np.std(reward_list)),
        "tvoc_delta_mean": float(np.mean(tvoc_delta_list))
        if tvoc_delta_list
        else np.nan,
        "tvoc_delta_std": float(np.std(tvoc_delta_list)) if tvoc_delta_list else np.nan,
        "mean_rh_excess": float(np.mean(rh_excess_list)) if rh_excess_list else np.nan,
        "mean_co2": float(np.mean(co2_list)) if co2_list else np.nan,
        "mean_temp": float(np.mean(temp_list)) if temp_list else np.nan,
        "tvoc_target_hit_rate": step_tvoc_hit / max(1, step_count),
        "co2_target_hit_rate": step_co2_hit / max(1, step_count),
        "action_entropy": action_entropy,
        **action_ratio,
    }
    return pd.DataFrame([out])


def tune_hyperparams(
    algo: AlgoName,
    base_sim: Pi1SpraySimulator,
    run_cfg: RLRunConfig,
    obs_divisors: dict,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = finalize_params(sample_hyperparams(algo, trial))
        gamma = float(params.get("gamma", 0.99))
        env_cfg = make_env_cfg(base_sim, gamma=gamma, obs_divisors=obs_divisors)

        train_env = build_vec_env(
            base_sim=base_sim,
            env_cfg=env_cfg,
            seed=run_cfg.seed + trial.number * 1000,
            n_envs=run_cfg.n_envs,
            normalize=True,
            use_subproc=run_cfg.use_subproc,
            gamma=gamma,
        )
        model = build_model(algo, train_env, params, seed=run_cfg.seed + trial.number)
        model.learn(total_timesteps=run_cfg.tune_budget)

        eval_env = make_eval_env(
            base_sim=base_sim,
            env_cfg=env_cfg,
            train_env=train_env,
            seed=run_cfg.seed + 50_000 + trial.number,
        )
        metrics = evaluate_model(
            model=model,
            eval_env=eval_env,
            env_cfg=env_cfg,
            n_episodes=max(5, run_cfg.eval_episodes // 2),
            seed=run_cfg.seed + 100_000 + trial.number,
        )

        train_env.close()
        eval_env.close()
        return float(metrics.loc[0, "reward_mean"])

    log_info(f"Tuning {algo.upper()} with {run_cfg.tune_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=run_cfg.tune_trials, show_progress_bar=True)
    best = finalize_params(study.best_params)
    log_info(f"Best {algo.upper()} params: {best}")
    return best


def train_one_budget(
    algo: AlgoName,
    base_sim: Pi1SpraySimulator,
    env_cfg: EnvConfig,
    params: dict,
    total_timesteps: int,
    run_cfg: RLRunConfig,
) -> tuple:
    log_info(f"Training {algo.upper()} for {total_timesteps:,} timesteps...")
    train_env = build_vec_env(
        base_sim=base_sim,
        env_cfg=env_cfg,
        seed=run_cfg.seed,
        n_envs=run_cfg.n_envs,
        normalize=True,
        use_subproc=run_cfg.use_subproc,
        gamma=params.get("gamma", 0.99),
    )
    model = build_model(algo, train_env, params, seed=run_cfg.seed)
    model.learn(total_timesteps=total_timesteps)

    eval_env = make_eval_env(
        base_sim=base_sim,
        env_cfg=env_cfg,
        train_env=train_env,
        seed=run_cfg.seed + 999,
    )
    metrics = evaluate_model(
        model=model,
        eval_env=eval_env,
        env_cfg=env_cfg,
        n_episodes=run_cfg.eval_episodes,
        seed=run_cfg.seed + 1_000,
    )
    metrics.insert(0, "algo", algo)
    metrics["timesteps"] = int(total_timesteps)

    train_env.close()
    eval_env.close()

    return model, metrics


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    log_info("=" * 80)
    log_info("IoT Farm RL Training Script (use_subproc=True)")
    log_info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info("=" * 80)

    # Load cleaned data
    data_file = BASE_DIR / "pi-1-data-clean-5min.csv"
    if not data_file.exists():
        log_info(f"ERROR: {data_file} not found. Exiting.")
        return

    log_info(f"Loading {data_file}...")
    pi1_df = pd.read_csv(data_file)
    log_info(f"  Loaded {len(pi1_df)} rows, {pi1_df.shape[1]} columns")

    # Compute observation divisors from data
    log_info("Computing observation normalization constants...")
    _obs_tvoc = pd.to_numeric(pi1_df["tvoc"], errors="coerce").dropna()
    _obs_co2 = pd.to_numeric(pi1_df["co2"], errors="coerce").dropna()
    _obs_temp = pd.to_numeric(pi1_df["temp"], errors="coerce").dropna()
    _obs_rh = pd.to_numeric(pi1_df["humidity"], errors="coerce").dropna()

    obs_divisors = {
        "tvoc": float(_obs_tvoc.quantile(0.99)),
        "co2": float(_obs_co2.quantile(0.99)),
        "temp": float(_obs_temp.quantile(0.99)),
        "rh_min": float(_obs_rh.quantile(0.01)),
        "rh_span": float(_obs_rh.quantile(0.99)) - float(_obs_rh.quantile(0.01)),
    }
    log_info(f"  TVOC divisor: {obs_divisors['tvoc']:.2f}")
    log_info(f"  CO2 divisor: {obs_divisors['co2']:.2f}")
    log_info(f"  TEMP divisor: {obs_divisors['temp']:.2f}")
    log_info(
        f"  RH min: {obs_divisors['rh_min']:.2f}, span: {obs_divisors['rh_span']:.2f}"
    )

    # Build simulator
    log_info("Building simulator...")
    sim = Pi1SpraySimulator(pi1_df, config=Pi1SimulatorConfig())
    log_info(f"  Simulator T={sim.T} steps")

    # Run config
    run_cfg = RLRunConfig(use_subproc=True)
    log_info(f"RL Config: n_envs={run_cfg.n_envs}, use_subproc={run_cfg.use_subproc}")
    log_info(
        f"  Tune: {run_cfg.tune_trials} trials, {run_cfg.tune_budget:,} steps/trial"
    )
    log_info(f"  Train budgets: {run_cfg.budgets}")
    log_info(f"  Eval episodes: {run_cfg.eval_episodes}")

    # Tune & train
    algorithms = ("ppo", "dqn")
    best_params_by_algo: dict[str, dict] = {}

    log_info("\n" + "=" * 80)
    log_info("HYPERPARAMETER TUNING PHASE")
    log_info("=" * 80)

    for algo in algorithms:
        best_params = tune_hyperparams(algo, sim, run_cfg, obs_divisors)
        best_params_by_algo[algo] = best_params

    # Train on multiple budgets
    log_info("\n" + "=" * 80)
    log_info("TRAINING & EVALUATION PHASE")
    log_info("=" * 80)

    all_results: list[pd.DataFrame] = []

    for algo in algorithms:
        best_params = best_params_by_algo[algo]
        gamma = float(best_params.get("gamma", 0.99))
        env_cfg = make_env_cfg(sim, gamma=gamma, obs_divisors=obs_divisors)

        for budget in run_cfg.budgets:
            log_info(f"\n[{algo.upper()}] Budget: {budget:,} timesteps")
            _, metrics = train_one_budget(
                algo=algo,
                base_sim=sim,
                env_cfg=env_cfg,
                params=best_params,
                total_timesteps=budget,
                run_cfg=run_cfg,
            )
            metrics.insert(0, "gamma", gamma)
            all_results.append(metrics)

            # Log metrics
            for col in metrics.columns:
                val = metrics.loc[0, col]
                if isinstance(val, (int, float)):
                    log_info(f"  {col}: {val:.4f}")
                else:
                    log_info(f"  {col}: {val}")

    # Save results
    results_df = pd.concat(all_results, ignore_index=True)
    results_df = results_df.sort_values(["algo", "timesteps"]).reset_index(drop=True)

    log_info("\n" + "=" * 80)
    log_info("FINAL RESULTS")
    log_info("=" * 80)
    log_info(f"\n{results_df.to_string(index=False)}")

    results_df.to_csv(RESULTS_FILE, index=False)
    log_info(f"\nResults saved to {RESULTS_FILE}")

    log_info("\n" + "=" * 80)
    log_info(f"Training complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"Log file: {LOG_FILE}")
    log_info("=" * 80)


if __name__ == "__main__":
    main()
