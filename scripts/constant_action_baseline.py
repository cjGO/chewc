#!/usr/bin/env python3
"""Compute constant-action baselines for the StoaEnv breeding environment.

This utility reuses the environment configuration from the PPO training loop
and evaluates simple constant-action policies.  For each supplied action value
it reports the terminal reward (mean TBV) and the final population phenotypic
mean aggregated over several stochastic replications.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from chewc.gym import StoaEnv
from chewc.pheno import calculate_phenotypes


@dataclass(frozen=True)
class EnvConfig:
    """Container mirroring the PPO setup defaults for StoaEnv."""

    total_gen: int = 10
    n_founders: int = 2
    n_pop_size: int = 100
    n_chr: int = 5
    n_loci: int = 100
    n_qtl_per_chr: int = 50
    max_crossovers: int = 10

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EnvConfig":
        return cls(
            total_gen=args.total_gen,
            n_founders=args.n_founders,
            n_pop_size=args.n_pop_size,
            n_chr=args.n_chr,
            n_loci=args.n_loci,
            n_qtl_per_chr=args.n_qtl_per_chr,
            max_crossovers=args.max_crossovers,
        )


def make_env(cfg: EnvConfig) -> StoaEnv:
    """Instantiate the environment with the desired settings."""
    return StoaEnv(
        n_founders=cfg.n_founders,
        n_pop_size=cfg.n_pop_size,
        n_chr=cfg.n_chr,
        n_loci=cfg.n_loci,
        n_qtl_per_chr=cfg.n_qtl_per_chr,
        total_gen=cfg.total_gen,
        max_crossovers=cfg.max_crossovers,
    )


def run_episode(
    env: StoaEnv,
    action_value: float,
    rng_key: jax.Array,
) -> Tuple[float, float, float]:
    """Simulate a single episode using a constant action."""
    params = env.default_params
    reset_key, rng_key = jax.random.split(rng_key)
    _, state = env.reset_env(reset_key, params)

    action = jnp.array([action_value], dtype=jnp.float32)
    final_reward = 0.0

    while True:
        rng_key, step_key = jax.random.split(rng_key)
        obs, next_state, reward, done, _ = env.step_env(step_key, state, action, params)
        final_reward = float(reward)
        state = next_state
        if bool(done):
            break

    # Recompute phenotypes on the final population for reporting.
    phenokey, _ = jax.random.split(state.key)
    phenotypes, tbv = calculate_phenotypes(
        phenokey,
        population=state.population,
        trait=env.trait_architecture,
        heritability=env.heritabilities,
    )

    phenotypes_np = np.asarray(phenotypes[:, 0])
    tbv_np = np.asarray(tbv[:, 0])
    final_pheno_mean = float(phenotypes_np.mean())
    final_tbv_mean = float(tbv_np.mean())
    return final_reward, final_pheno_mean, final_tbv_mean


def evaluate_constant_actions(
    env: StoaEnv,
    action_values: Sequence[float],
    num_episodes: int,
    seed: int,
) -> List[dict]:
    """Evaluate constant-action policies and aggregate summary statistics."""
    rng = jax.random.PRNGKey(seed)
    results: List[dict] = []

    for action_value in action_values:
        rewards: List[float] = []
        pheno_means: List[float] = []
        tbv_means: List[float] = []

        for _ in range(num_episodes):
            rng, episode_key = jax.random.split(rng)
            reward, pheno_mean, tbv_mean = run_episode(env, action_value, episode_key)
            rewards.append(reward)
            pheno_means.append(pheno_mean)
            tbv_means.append(tbv_mean)

        rewards_np = np.asarray(rewards, dtype=np.float32)
        pheno_np = np.asarray(pheno_means, dtype=np.float32)
        tbv_np = np.asarray(tbv_means, dtype=np.float32)

        results.append(
            {
                "action": float(action_value),
                "reward_mean": float(rewards_np.mean()),
                "reward_std": float(rewards_np.std(ddof=1)) if num_episodes > 1 else 0.0,
                "pheno_mean": float(pheno_np.mean()),
                "pheno_std": float(pheno_np.std(ddof=1)) if num_episodes > 1 else 0.0,
                "tbv_mean": float(tbv_np.mean()),
                "tbv_std": float(tbv_np.std(ddof=1)) if num_episodes > 1 else 0.0,
            }
        )

    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute constant-action baselines for the StoaEnv."
    )
    parser.add_argument(
        "--actions",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
        help="Constant action values to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=128,
        help="Number of stochastic replications per action (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base PRNG seed (default: %(default)s)."
    )
    parser.add_argument(
        "--total-gen",
        type=int,
        default=10,
        dest="total_gen",
        help="Number of generations per episode (default: %(default)s).",
    )
    parser.add_argument(
        "--n-founders",
        type=int,
        default=2,
        dest="n_founders",
        help="Founder population size (default: %(default)s).",
    )
    parser.add_argument(
        "--n-pop-size",
        type=int,
        default=100,
        dest="n_pop_size",
        help="Population size maintained after burn-in (default: %(default)s).",
    )
    parser.add_argument(
        "--n-chr",
        type=int,
        default=5,
        help="Number of chromosomes in the simulation (default: %(default)s).",
    )
    parser.add_argument(
        "--n-loci",
        type=int,
        default=100,
        help="Number of loci per chromosome (default: %(default)s).",
    )
    parser.add_argument(
        "--n-qtl-per-chr",
        type=int,
        default=50,
        dest="n_qtl_per_chr",
        help="Number of QTL sampled per chromosome (default: %(default)s).",
    )
    parser.add_argument(
        "--max-crossovers",
        type=int,
        default=10,
        dest="max_crossovers",
        help="Maximum number of crossovers per meiosis (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = EnvConfig.from_args(args)
    env = make_env(cfg)

    print("Environment configuration:")
    for field, value in asdict(cfg).items():
        print(f"  {field}: {value}")
    print(f"Evaluating {len(args.actions)} constant actions "
          f"over {args.episodes} episodes each...\n")

    results = evaluate_constant_actions(env, args.actions, args.episodes, args.seed)

    header = (
        "action",
        "reward_mean",
        "reward_std",
        "pheno_mean",
        "pheno_std",
        "tbv_mean",
        "tbv_std",
    )
    print(" | ".join(f"{name:>12}" for name in header))
    print("-" * (len(header) * 15))
    for row in results:
        print(
            " | ".join(
                [
                    f"{row['action']:12.3f}",
                    f"{row['reward_mean']:12.3f}",
                    f"{row['reward_std']:12.3f}",
                    f"{row['pheno_mean']:12.3f}",
                    f"{row['pheno_std']:12.3f}",
                    f"{row['tbv_mean']:12.3f}",
                    f"{row['tbv_std']:12.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
