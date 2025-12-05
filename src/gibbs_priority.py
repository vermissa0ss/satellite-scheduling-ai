import random

import numpy as np


class GibbsPriorityModel:
    """Priority/seed model using Gibbs sampling marginals from the hard-core model."""

    def __init__(self, beta=0.5, sweeps=30, burn=10, seed=0):
        if sweeps <= 0:
            raise ValueError("GibbsPriorityModel requires sweeps > 0")
        if burn < 0:
            raise ValueError("GibbsPriorityModel burn must be >= 0")
        self.beta = beta
        self.sweeps = sweeps
        self.burn = min(burn, sweeps - 1) if sweeps > 1 else 0
        self.rng_seed = seed
        self._state_key = "_gibbs_priority_state"

    def _ensure_state(self, component):
        state = getattr(component, self._state_key, None)
        if state is not None:
            return state
        state = self._run_sampler(component)
        setattr(component, self._state_key, state)
        return state

    def _run_sampler(self, component):
        n = component.n
        rng = random.Random(self.rng_seed + int(getattr(component, "comp_idx", 0)))
        lam = np.exp(self.beta * component.weights)
        x_mask = 0
        counts = np.zeros(n, dtype=np.int64)
        order = list(range(n))
        sweeps = self.sweeps
        burn = self.burn

        for sweep in range(sweeps):
            rng.shuffle(order)
            for v in order:
                bit = 1 << v
                if x_mask & bit:
                    x_mask &= ~bit
                if x_mask & component.neighbor_masks[v]:
                    continue
                prob_on = float(lam[v] / (1.0 + lam[v]))
                if rng.random() < prob_on:
                    x_mask |= bit
            if sweep >= burn:
                mask = x_mask
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    mask &= mask - 1

        kept = max(sweeps - burn, 1)
        marginals = counts.astype(np.float64) / float(kept)
        if marginals.max(initial=0.0) == 0.0:
            marginals = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
        order_by_marg = tuple(int(i) for i in np.argsort(-marginals, kind="mergesort"))
        seed_weight, seed_mask = component.greedy_from_order(order_by_marg, component.all_mask)
        return {
            "marginals": marginals,
            "seed_weight": float(seed_weight),
            "seed_mask": int(seed_mask),
        }

    def score(self, component, cand_mask):
        return self._ensure_state(component)["marginals"]

    def seed(self, component):
        state = self._ensure_state(component)
        return state["seed_weight"], state["seed_mask"]
