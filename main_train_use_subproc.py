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

TVOC_EXCESS_NORM = 20.0
THI_EXCESS_NORM = 0.040
RH_EXCESS_NORM = 6.25
COLD_EXCESS_NORM = 0.040
TVOC_PROGRESS_NORM = 5.0

TVOC_OBS_DIVISOR = 1.0
CO2_OBS_DIVISOR = 1.0
TEMP_OBS_DIVISOR = 1.0
RH_OBS_MIN = 0.0
RH_OBS_SPAN = 1.0

AlgoName = Literal["ppo", "dqn"]


@dataclass(frozen=True)
class RewardWeights:
    tvoc_excess_scale: float = 1.0
    use_tvoc_delta: bool = True
    tvoc_target: float = 50.0

    thi_scale: float = 0.45
    rh_excess_scale: float = 0.15
    rh_target: float = 90.0
    cold_scale: float = 0.30

    action_scale: float = 1.0
    action_curve_power: float = 3
    action_urgency_scale: float = 1.0


@dataclass
class Pi1SimulatorConfig:
    gamma: float = 0.99
    baseline_decay_frac_per_step: float = 0.40
    time_step_minutes: float = 5.0
    spray_volume_ml: float = 1000.0
    spray_mass_kg: float = 1.0
    room_area_m2: float = 400.0
    room_height_m: float = 3.0
    room_volume_m3: float = room_area_m2 * room_height_m
    room_total_air_mass_kg: float = room_volume_m3 * 1.176
    atmospheric_pressure_pa: float = 101325.0
    cp_air: float = 1.005
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
    tvoc_norm_divisor: float = field(default_factory=lambda: TVOC_OBS_DIVISOR)
    co2_norm_divisor: float = field(default_factory=lambda: CO2_OBS_DIVISOR)
    temp_norm_divisor: float = field(default_factory=lambda: TEMP_OBS_DIVISOR)
    rh_min: float = field(default_factory=lambda: RH_OBS_MIN)
    rh_span: float = field(default_factory=lambda: RH_OBS_SPAN)


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


def tetens_saturation_vapor_pressure(temp_c: float) -> float:
    """Calculate saturation vapor pressure in hPa using Tetens formula."""

    return 6.1078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def tvoc_spray_loss(dose_mg_L: float) -> float:
    """Simple model for TVOC reduction from spray dose."""

    beta = -np.log(1 - 0.649) / 1500.0
    return 1 - np.exp(-beta * dose_mg_L)


class Pi1SpraySimulator:
    """Hybrid reduced-order simulator for RL prototyping on pi-1 5-minute data."""

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
        tvoc_base = self.real_tvoc[t + 1]
        beta = -np.log(1 - 0.649) / 1500.0
        tvoc_next = (
            tvoc_base
            - (1 - np.exp(-beta * action)) * tvoc_base
            + self.rng.normal(0.0, self.cfg.tvoc_noise_sigma)
        )

        co2_base = self.real_co2[t + 1]
        co2_next = co2_base + self.rng.normal(0.0, self.cfg.co2_noise_sigma)
        co2_next = max(300.0, co2_next)

        if action == 0:
            temp_next = self.real_temp[t + 1] + self.rng.normal(
                0.0, self.cfg.temp_noise_sigma
            )
            rh_next = self.real_rh[t + 1] + self.rng.normal(
                0.0, self.cfg.rh_noise_sigma
            )
            rh_next = np.clip(rh_next, 0.0, 100.0)
            return np.array([tvoc_next, co2_next, temp_next, rh_next], dtype=float)

        rh_base = self.real_rh[t + 1]
        temp_base = self.real_temp[t + 1]

        wetbulb = wetbulb_stull(temp_base, rh_base)
        L = 2501 - (2.369 * temp_base)
        temp_drop = (self.cfg.spray_mass_kg * L) / (
            self.cfg.room_total_air_mass_kg * self.cfg.cp_air
        )
        temp_next = max(
            wetbulb,
            temp_base - temp_drop + self.rng.normal(0.0, self.cfg.temp_noise_sigma),
        )

        actual_liquid_evaporation = (
            (temp_base - temp_next) * self.cfg.room_total_air_mass_kg * self.cfg.cp_air
        ) / L

        e_s_initial = tetens_saturation_vapor_pressure(temp_base)
        e_initial = e_s_initial * (rh_base / 100.0)

        w_initial = 0.622 * (
            e_initial / (self.cfg.atmospheric_pressure_pa / 100.0 - e_initial)
        )

        w_added = actual_liquid_evaporation / self.cfg.room_total_air_mass_kg
        w_final = w_initial + w_added

        e_final = (w_final * self.cfg.atmospheric_pressure_pa / 100.0) / (
            0.622 + w_final
        )

        e_s_final = tetens_saturation_vapor_pressure(temp_next)

        rh_next = (e_final / e_s_final) * 100.0 + self.rng.normal(
            0.0, self.cfg.rh_noise_sigma
        )

        rh_next = min(rh_next, 100.0)
        if rh_next >= 99.5:
            rh_next = 100.0

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

        base_reward = tvoc_excess_term + thi_term + rh_term + cold_term + action_term

        phi_prev = self._phi(tvoc_prev, co2_prev, temp_prev, rh_prev, w)
        phi_next = self._phi(tvoc_next, co2_next, temp_next, rh_next, w)
        shaping = 0.0 if is_terminal else (self.cfg.gamma * phi_next - phi_prev)

        return float(base_reward + shaping)

    def _phi(
        self, tvoc: float, co2: float, temp: float, rh: float, w: RewardWeights
    ) -> float:
        thi = (0.8 * temp) + ((rh / 100.0) * (temp - 14.4)) + 46.4

        tvoc_excess = max(0.0, tvoc - w.tvoc_target) / TVOC_EXCESS_NORM
        thi_excess = max(0.0, thi - 73.0) / THI_EXCESS_NORM
        rh_excess = max(0.0, rh - w.rh_target) / RH_EXCESS_NORM
        cold_excess = max(0.0, 18.0 - temp) / COLD_EXCESS_NORM

        return -(
            w.tvoc_excess_scale * tvoc_excess
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
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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

        obs = self._build_obs(self.state)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        dose = self.action_to_concentration[int(action)]
        state_prev = self.state.copy()
        next_state = self.sim.step(state_prev, dose, self.t).astype(np.float32)

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
        }

        obs = self._build_obs(self.state)
        return obs, reward, terminated, truncated, info

    def _build_obs(self, s: np.ndarray) -> np.ndarray:
        return np.array(
            [
                s[0] / self.cfg.tvoc_norm_divisor,
                s[1] / self.cfg.co2_norm_divisor,
                s[2] / self.cfg.temp_norm_divisor,
                (s[3] - self.cfg.rh_min) / self.cfg.rh_span,
            ],
            dtype=np.float32,
        )


# ==============================================================================
# RL INFRASTRUCTURE
# ==============================================================================

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN}


def make_env_cfg(sim: Pi1SpraySimulator, gamma: float) -> EnvConfig:
    return EnvConfig(
        reward=sim.cfg.reward,
        max_steps=250,
        gamma=gamma,
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
    step_count = 0

    for ep in range(n_episodes):
        eval_env.seed(seed + ep)
        obs = eval_env.reset()
        tvoc_start = float(eval_env.get_attr("state")[0][0])
        ep_reward = 0.0
        done = [False]

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = eval_env.step(action)

            ep_reward += float(rewards[0])
            info = infos[0]

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
    dominant_action_ratio = float(action_probs.max())

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
        "action_entropy": action_entropy,
        "dominant_action_ratio": dominant_action_ratio,
        **action_ratio,
    }
    return pd.DataFrame([out])


def tune_hyperparams(
    algo: AlgoName,
    base_sim: Pi1SpraySimulator,
    run_cfg: RLRunConfig,
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = finalize_params(sample_hyperparams(algo, trial))
        gamma = float(params.get("gamma", 0.99))
        env_cfg = make_env_cfg(base_sim, gamma=gamma)

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

    # Load cleaned data for calibration and training.
    cleaned_files = [
        BASE_DIR / "pi-1-data-clean-5min.csv",
        BASE_DIR / "pi-2-data-clean-5min.csv",
    ]
    cleaned_frames: dict[str, pd.DataFrame] = {}
    for data_file in cleaned_files:
        if not data_file.exists():
            log_info(f"ERROR: {data_file} not found. Exiting.")
            return

        log_info(f"Loading {data_file}...")
        cleaned_df = pd.read_csv(data_file)
        cleaned_frames[data_file.name] = cleaned_df
        log_info(f"  Loaded {len(cleaned_df)} rows, {cleaned_df.shape[1]} columns")

    pi1_df = cleaned_frames["pi-1-data-clean-5min.csv"].copy()
    cal_df = pd.concat(list(cleaned_frames.values()), ignore_index=True)

    # Compute observation normalization constants from the cleaned calibration data.
    log_info("Computing observation normalization constants...")
    _obs_tvoc = pd.to_numeric(cal_df["tvoc"], errors="coerce").dropna()
    _obs_co2 = pd.to_numeric(cal_df["co2"], errors="coerce").dropna()
    _obs_temp = pd.to_numeric(cal_df["temp"], errors="coerce").dropna()
    _obs_rh = pd.to_numeric(cal_df["humidity"], errors="coerce").dropna()

    global TVOC_OBS_DIVISOR, CO2_OBS_DIVISOR, TEMP_OBS_DIVISOR, RH_OBS_MIN, RH_OBS_SPAN
    TVOC_OBS_DIVISOR = float(_obs_tvoc.quantile(0.99))
    CO2_OBS_DIVISOR = float(_obs_co2.quantile(0.99))
    TEMP_OBS_DIVISOR = float(_obs_temp.quantile(0.99))
    RH_OBS_MIN = float(_obs_rh.quantile(0.01))
    RH_OBS_SPAN = float(_obs_rh.quantile(0.99)) - RH_OBS_MIN

    log_info(f"  TVOC divisor: {TVOC_OBS_DIVISOR:.2f}")
    log_info(f"  CO2 divisor: {CO2_OBS_DIVISOR:.2f}")
    log_info(f"  TEMP divisor: {TEMP_OBS_DIVISOR:.2f}")
    log_info(f"  RH min: {RH_OBS_MIN:.2f}, span: {RH_OBS_SPAN:.2f}")

    # Reward normalizers from the cleaned calibration data.
    log_info("Computing reward normalization constants...")

    def one_step_abs_diff(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce").dropna()
        return x.diff().abs().dropna()

    def positive_excess(x: pd.Series, target: float) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce").dropna()
        return (x - target).clip(lower=0.0)

    global \
        TVOC_PROGRESS_NORM, \
        TVOC_EXCESS_NORM, \
        THI_EXCESS_NORM, \
        RH_EXCESS_NORM, \
        COLD_EXCESS_NORM
    TVOC_PROGRESS_NORM = float(one_step_abs_diff(cal_df["tvoc"]).median())
    TVOC_EXCESS_NORM = float(positive_excess(cal_df["tvoc"], 50.0).quantile(0.75))
    THI_EXCESS_NORM = float(
        positive_excess(
            (0.8 * cal_df["temp"])
            + ((cal_df["humidity"] / 100.0) * (cal_df["temp"] - 14.4))
            + 46.4,
            73.0,
        ).quantile(0.75)
    )
    RH_EXCESS_NORM = float(positive_excess(cal_df["humidity"], 90.0).quantile(0.75))
    COLD_EXCESS_NORM = float(positive_excess(18.0 - cal_df["temp"], 0.0).quantile(0.75))

    log_info(f"  TVOC_PROGRESS_NORM: {TVOC_PROGRESS_NORM:.4f}")
    log_info(f"  TVOC_EXCESS_NORM: {TVOC_EXCESS_NORM:.4f}")
    log_info(f"  THI_EXCESS_NORM: {THI_EXCESS_NORM:.4f}")
    log_info(f"  RH_EXCESS_NORM: {RH_EXCESS_NORM:.4f}")
    log_info(f"  COLD_EXCESS_NORM: {COLD_EXCESS_NORM:.4f}")

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
        best_params = tune_hyperparams(algo, sim, run_cfg)
        best_params_by_algo[algo] = best_params

    # Train on multiple budgets
    log_info("\n" + "=" * 80)
    log_info("TRAINING & EVALUATION PHASE")
    log_info("=" * 80)

    all_results: list[pd.DataFrame] = []

    for algo in algorithms:
        best_params = best_params_by_algo[algo]
        gamma = float(best_params.get("gamma", 0.99))
        env_cfg = make_env_cfg(sim, gamma=gamma)

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
