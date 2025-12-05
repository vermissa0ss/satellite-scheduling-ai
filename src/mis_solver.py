import argparse
import heapq
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


sys.setrecursionlimit(100000)

FROZEN_ROOT = Path("frozen")


def load_dataset(dataset_id):
    root = FROZEN_ROOT / dataset_id
    if not root.exists():
        raise FileNotFoundError(f"Frozen dataset folder missing: {root}")
    data = np.load(root / "data.npz")
    meta = json.loads((root / "meta.json").read_text())
    return data, meta


class ComponentView:
    def __init__(self, data, meta, comp_idx):
        self.data = data
        self.meta = meta

        comp_ptr = data["comp_ptr"]
        comp_order = data["comp_order"]
        self.comp_idx = comp_idx
        self.start = int(comp_ptr[comp_idx])
        self.end = int(comp_ptr[comp_idx + 1])
        self.global_vertices = comp_order[self.start : self.end].astype(np.int64, copy=False)
        self.n = int(self.global_vertices.size)
        if self.n == 0:
            raise ValueError("Empty component encountered")

        self.indptr = data["indptr"]
        self.indices = data["indices"]
        w_all = data["w"]
        dwell_all = data["dwell"]
        tend_all = data["t_end"]
        degree_all = data["degree"]
        self.metis_to_rec = data["metis_to_rec"]

        self.weights = np.ascontiguousarray(w_all[self.global_vertices])
        self.dwell = np.ascontiguousarray(dwell_all[self.global_vertices])
        self.t_end = np.ascontiguousarray(tend_all[self.global_vertices])
        self.degree = np.ascontiguousarray(degree_all[self.global_vertices])

        self.local_index = {int(g): i for i, g in enumerate(self.global_vertices)}
        self._build_neighbor_masks()

        self.order_heavy = self._map_global_order(data["order_heaviest"])
        self.order_degree = self._map_global_order(data["order_degree"])
        self.order_deadline = self._map_global_order(data["order_deadline"])
        self.order_hybrid = self._map_global_order(data["order_hybrid"])

        density = self.weights / np.maximum(self.dwell, 1e-6)
        self.order_density = tuple(int(i) for i in np.argsort(-density, kind="mergesort"))

        self.greedy_orders = {
            "density": self.order_density,
            "heaviest": self.order_heavy,
            "degree": self.order_degree,
            "deadline": self.order_deadline,
            "hybrid": self.order_hybrid,
        }
        self.all_mask = (1 << self.n) - 1
        self._priority_model = None

    def _map_global_order(self, global_order):
        mapped = []
        for g in global_order:
            li = self.local_index.get(int(g))
            if li is not None:
                mapped.append(li)
            if len(mapped) == self.n:
                break
        return tuple(mapped)

    def _build_neighbor_masks(self):
        masks = [0] * self.n
        indptr = self.indptr
        indices = self.indices
        local_index = self.local_index
        for local_i, g in enumerate(self.global_vertices):
            start = indptr[g]
            end = indptr[g + 1]
            mask = 0
            for nb in indices[start:end]:
                li = local_index.get(int(nb))
                if li is not None:
                    mask |= 1 << li
            masks[local_i] = mask
        self.neighbor_masks = masks

    def iter_order(self, order, cand_mask):
        remaining = cand_mask
        for v in order:
            bit = 1 << v
            if remaining & bit:
                yield v
                remaining &= ~bit
            if not remaining:
                break

    def attach_priority_model(self, model):
        self._priority_model = model

    def get_priority_model(self):
        return getattr(self, "_priority_model", None)

    def _order_by_scores(self, scores, cand_mask):
        items = []
        remaining = cand_mask
        while remaining:
            bit = remaining & -remaining
            v = (bit.bit_length() - 1)
            items.append((float(scores[v]), v))
            remaining &= remaining - 1
        items.sort(key=lambda x: x[0], reverse=True)
        return tuple(v for _, v in items)

    def pick_branch_vertex(self, cand_mask):
        model = self.get_priority_model()
        if model is not None:
            scores = model.score(self, cand_mask)
            best_vertex = -1
            best_score = -math.inf
            remaining = cand_mask
            while remaining:
                bit = remaining & -remaining
                v = (bit.bit_length() - 1)
                score = float(scores[v])
                if score > best_score:
                    best_score = score
                    best_vertex = v
                remaining &= remaining - 1
            if best_vertex >= 0:
                return best_vertex
        for v in self.order_hybrid:
            if cand_mask & (1 << v):
                return v
        # fallback: take lowest-set bit
        lsb = cand_mask & -cand_mask
        return (lsb.bit_length() - 1)

    def greedy_clique_cover(self, ordered_vertices, cand_mask):
        cliques = []
        maxima = []
        full_mask = self.all_mask
        for v in ordered_vertices:
            bit = 1 << v
            if not (cand_mask & bit):
                continue
            placed = False
            neigh = self.neighbor_masks[v]
            nn_mask = (~neigh) & full_mask
            for idx, cmask in enumerate(cliques):
                if cmask & nn_mask:
                    continue
                cliques[idx] |= bit
                if self.weights[v] > maxima[idx]:
                    maxima[idx] = self.weights[v]
                placed = True
                break
            if not placed:
                cliques.append(bit)
                maxima.append(self.weights[v])
        return float(sum(maxima))

    def upper_bound(self, cand_mask):
        if cand_mask == 0:
            return 0.0
        order1 = tuple(self.iter_order(self.order_heavy, cand_mask))
        ub = self.greedy_clique_cover(order1, cand_mask)
        order2 = tuple(self.iter_order(self.order_degree, cand_mask))
        ub = min(ub, self.greedy_clique_cover(order2, cand_mask))
        model = self.get_priority_model()
        if model is not None:
            scores = model.score(self, cand_mask)
            order_nn = self._order_by_scores(scores, cand_mask)
            ub = min(ub, self.greedy_clique_cover(order_nn, cand_mask))
        return ub

    def greedy_lower_bound(self, cand_mask):
        model = self.get_priority_model()
        if model is not None:
            if cand_mask == self.all_mask and hasattr(model, "seed"):
                return model.seed(self)
            scores = model.score(self, cand_mask)
            order_nn = self._order_by_scores(scores, cand_mask)
            return self.greedy_from_order(order_nn, cand_mask)
        return self.greedy_from_order(self.order_density, cand_mask)

    def greedy_from_order(self, order, cand_mask=None):
        mask = self.all_mask if cand_mask is None else cand_mask
        total = 0.0
        selection = 0
        for v in self.iter_order(order, mask):
            bit = 1 << v
            if not (mask & bit):
                continue
            total += self.weights[v]
            selection |= bit
            mask &= ~bit
            mask &= ~self.neighbor_masks[v]
            if mask == 0:
                break
        return total, selection

    def exact_bruteforce_weight(self, subset_vertices):
        k = len(subset_vertices)
        if k > 22:
            raise ValueError("Subset too large for brute-force exact check")
        neighbor_masks = []
        weights = []
        for i, v in enumerate(subset_vertices):
            weights.append(self.weights[v])
            mask = 0
            neighbor_bits = self.neighbor_masks[v]
            for j, u in enumerate(subset_vertices):
                if neighbor_bits & (1 << u):
                    mask |= 1 << j
            neighbor_masks.append(mask)
        best = 0.0
        for choice in range(1 << k):
            valid = True
            total = 0.0
            for i in range(k):
                if not (choice & (1 << i)):
                    continue
                if neighbor_masks[i] & choice:
                    valid = False
                    break
                total += weights[i]
            if valid and total > best:
                best = total
        return best

    def validate_upper_bounds(self, samples=10, subset_size=14, rng=None):
        rng = rng or random.Random(0)
        for _ in range(samples):
            if self.n <= subset_size:
                subset = list(range(self.n))
            else:
                subset = rng.sample(range(self.n), subset_size)
            subset_mask = 0
            for v in subset:
                subset_mask |= 1 << v
            ub = self.upper_bound(subset_mask)
            exact = self.exact_bruteforce_weight(subset)
            if ub + 1e-9 < exact:
                raise AssertionError(
                    f"Upper bound {ub} underestimated exact {exact} on subset {subset}"
                )


class BranchAndBoundSolver:
    def __init__(self, component, time_limit=None, node_limit=None):
        self.component = component
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.best_weight = 0.0
        self.best_mask = 0
        self.nodes = 0
        self.anytime = []
        self.start = time.time()
        self.interrupted = False

    def solve(self):
        lb, sel = self.component.greedy_lower_bound(self.component.all_mask)
        self.best_weight = lb
        self.best_mask = sel
        self.anytime.append((0.0, self.best_weight))
        self._dfs(self.component.all_mask, 0.0, 0)
        duration = time.time() - self.start
        if not self.anytime or abs(self.anytime[-1][0] - duration) > 1e-9:
            self.anytime.append((duration, self.best_weight))
        return {
            "best_weight": self.best_weight,
            "best_mask": self.best_mask,
            "nodes": self.nodes,
            "time": duration,
            "anytime": self.anytime,
            "interrupted": self.interrupted,
        }

    def _dfs(self, cand_mask, current_weight, current_mask):
        if self.interrupted:
            return
        if self.time_limit and (time.time() - self.start) > self.time_limit:
            self.interrupted = True
            return
        if self.node_limit and self.nodes >= self.node_limit:
            self.interrupted = True
            return
        self.nodes += 1

        ub = current_weight + self.component.upper_bound(cand_mask)
        if ub <= self.best_weight + 1e-9:
            return

        if cand_mask == 0:
            if current_weight > self.best_weight + 1e-9:
                self.best_weight = current_weight
                self.best_mask = current_mask
                self.anytime.append((time.time() - self.start, self.best_weight))
            return

        v = self.component.pick_branch_vertex(cand_mask)
        bit = 1 << v

        # Include branch
        include_mask = cand_mask & ~bit
        include_mask &= ~self.component.neighbor_masks[v]
        self._dfs(
            include_mask,
            current_weight + self.component.weights[v],
            current_mask | bit,
        )

        # Exclude branch
        exclude_mask = cand_mask & ~bit
        self._dfs(exclude_mask, current_weight, current_mask)


class AStarSolver:
    def __init__(self, component, time_limit=None, node_limit=None):
        self.component = component
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.start = time.time()
        self.nodes = 0
        self.anytime = []
        self.interrupted = False

    def solve(self):
        lb, sel = self.component.greedy_lower_bound(self.component.all_mask)
        best_weight = lb
        best_mask = sel
        self.anytime.append((0.0, best_weight))

        pq = []
        initial_cand = self.component.all_mask
        initial_ub = self.component.upper_bound(initial_cand)
        heapq.heappush(
            pq,
            (-(initial_ub), -0.0, initial_cand, 0.0, 0),
        )

        while pq and not self.interrupted:
            if self.time_limit and (time.time() - self.start) > self.time_limit:
                self.interrupted = True
                break
            if self.node_limit and self.nodes >= self.node_limit:
                self.interrupted = True
                break

            f_neg, g_neg, cand_mask, current_weight, sel_mask = heapq.heappop(pq)
            self.nodes += 1
            if current_weight + self.component.upper_bound(cand_mask) <= best_weight + 1e-9:
                continue
            if cand_mask == 0:
                if current_weight > best_weight + 1e-9:
                    best_weight = current_weight
                    best_mask = sel_mask
                    self.anytime.append((time.time() - self.start, best_weight))
                continue

            v = self.component.pick_branch_vertex(cand_mask)
            bit = 1 << v

            include_mask = cand_mask & ~bit
            include_mask &= ~self.component.neighbor_masks[v]
            weight_inc = current_weight + self.component.weights[v]
            ub_inc = self.component.upper_bound(include_mask)
            if ub_inc + weight_inc > best_weight + 1e-9:
                heapq.heappush(
                    pq,
                    (-(weight_inc + ub_inc), -weight_inc, include_mask, weight_inc, sel_mask | bit),
                )

            exclude_mask = cand_mask & ~bit
            ub_exc = self.component.upper_bound(exclude_mask)
            if ub_exc + current_weight > best_weight + 1e-9:
                heapq.heappush(
                    pq,
                    (-(current_weight + ub_exc), -current_weight, exclude_mask, current_weight, sel_mask),
                )

        duration = time.time() - self.start
        if not self.anytime or abs(self.anytime[-1][0] - duration) > 1e-9:
            self.anytime.append((duration, best_weight))
        return {
            "best_weight": best_weight,
            "best_mask": best_mask,
            "nodes": self.nodes,
            "time": duration,
            "anytime": self.anytime,
            "interrupted": self.interrupted,
        }


class WeightedBestFirstSolver:
    def __init__(self, component, eps_schedule=None, time_limit=None, node_limit=None):
        self.component = component
        if not eps_schedule:
            eps_schedule = (1.5, 1.25, 1.0)
        self.eps_schedule = tuple(float(max(1.0, eps)) for eps in eps_schedule)
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.nodes = 0
        self.anytime = []
        self.interrupted = False
        self.start = None

    def _push(self, pq, eps, cand_mask, g_weight, sel_mask, incumbent):
        ub = self.component.upper_bound(cand_mask)
        if g_weight + ub <= incumbent + 1e-9:
            return
        f_score = g_weight + eps * ub
        heapq.heappush(pq, (-(f_score), -g_weight, cand_mask, g_weight, sel_mask))

    def _run_pass(self, eps, deadline, incumbent, best_mask):
        pq = []
        self._push(pq, eps, self.component.all_mask, 0.0, 0, incumbent)

        while pq and not self.interrupted:
            now = time.time()
            if self.time_limit and (now - self.start) > self.time_limit:
                self.interrupted = True
                break
            if deadline and now > deadline:
                break
            if self.node_limit and self.nodes >= self.node_limit:
                self.interrupted = True
                break

            _, g_neg, cand_mask, g_weight, sel_mask = heapq.heappop(pq)
            self.nodes += 1
            ub_true = self.component.upper_bound(cand_mask)
            if g_weight + ub_true <= incumbent + 1e-9:
                continue

            if cand_mask == 0:
                if g_weight > incumbent + 1e-9:
                    incumbent = g_weight
                    best_mask = sel_mask
                    self.anytime.append((time.time() - self.start, incumbent))
                continue

            v = self.component.pick_branch_vertex(cand_mask)
            bit = 1 << v

            include_mask = cand_mask & ~bit
            include_mask &= ~self.component.neighbor_masks[v]
            self._push(
                pq,
                eps,
                include_mask,
                g_weight + self.component.weights[v],
                sel_mask | bit,
                incumbent,
            )

            exclude_mask = cand_mask & ~bit
            self._push(pq, eps, exclude_mask, g_weight, sel_mask, incumbent)

        return incumbent, best_mask

    def solve(self):
        self.start = time.time()
        lb, sel = self.component.greedy_lower_bound(self.component.all_mask)
        incumbent = lb
        best_mask = sel
        self.anytime.append((0.0, incumbent))

        deadlines = None
        if self.time_limit:
            slice_len = self.time_limit / max(len(self.eps_schedule), 1)
            deadlines = [self.start + (i + 1) * slice_len for i in range(len(self.eps_schedule))]
        for idx, eps in enumerate(self.eps_schedule):
            deadline = deadlines[idx] if deadlines else None
            incumbent, best_mask = self._run_pass(eps, deadline, incumbent, best_mask)
            if self.interrupted:
                break

        duration = time.time() - self.start
        if not self.anytime or abs(self.anytime[-1][0] - duration) > 1e-9:
            self.anytime.append((duration, incumbent))
        return {
            "best_weight": incumbent,
            "best_mask": best_mask,
            "nodes": self.nodes,
            "time": duration,
            "anytime": self.anytime,
            "interrupted": self.interrupted,
        }


class MultiStartLocalSearchSolver:
    """
    Pure heuristic MIS solver (no optimality guarantee).

    Runs multiple randomized greedy passes using a degree-aware score and
    applies a simple 1-swap local improvement to each pass. Keeps the best
    independent set found within the time budget.
    """

    def __init__(self, component, restarts=16, time_limit=None, rng_seed=0, alpha=0.5):
        self.component = component
        self.restarts = int(max(1, restarts))
        self.time_limit = time_limit
        self.rng = random.Random(rng_seed + int(getattr(component, "comp_idx", 0)))
        self.alpha = float(alpha)
        self.anytime = []
        self.start = None
        self.interrupted = False

    def _time_exceeded(self):
        return self.time_limit is not None and (time.time() - self.start) > self.time_limit

    def _base_scores(self):
        w = self.component.weights
        dwell = np.maximum(self.component.dwell, 1.0)
        deg = self.component.degree.astype(np.float64, copy=False)
        base = w / dwell
        max_deg = float(deg.max()) if deg.size else 1.0
        norm_deg = deg / max_deg
        return base - self.alpha * norm_deg

    def _randomized_order(self, scores):
        items = list(range(self.component.n))
        items.sort(key=lambda v: (-float(scores[v]), self.rng.random()))
        return tuple(items)

    def _local_1swap_improve(self, sel_mask, weight):
        """
        Try to insert a vertex and drop its selected neighbors if that is a net gain.
        """
        weights = self.component.weights
        neighbor_masks = self.component.neighbor_masks
        improved = True
        while improved and not self._time_exceeded():
            improved = False
            best_delta = 0.0
            best_v = -1
            best_drop_mask = 0

            remaining = (~sel_mask) & self.component.all_mask
            while remaining:
                bit = remaining & -remaining
                v = bit.bit_length() - 1
                remaining &= remaining - 1

                nb_mask = neighbor_masks[v] & sel_mask
                if nb_mask == 0:
                    delta = float(weights[v])
                else:
                    drop_w = 0.0
                    tmp = nb_mask
                    while tmp:
                        b2 = tmp & -tmp
                        u = b2.bit_length() - 1
                        drop_w += float(weights[u])
                        tmp &= tmp - 1
                    delta = float(weights[v]) - drop_w

                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_v = v
                    best_drop_mask = nb_mask

            if best_delta > 1e-9 and best_v >= 0:
                sel_mask &= ~best_drop_mask
                sel_mask |= 1 << best_v
                weight += best_delta
                improved = True

        if self._time_exceeded():
            self.interrupted = True
        return weight, sel_mask

    def solve(self):
        self.start = time.time()
        best_weight = 0.0
        best_mask = 0
        self.anytime.append((0.0, best_weight))

        scores = self._base_scores()
        for _ in range(self.restarts):
            if self._time_exceeded():
                self.interrupted = True
                break

            order = self._randomized_order(scores)
            greedy_w, greedy_mask = self.component.greedy_from_order(order)

            if not self._time_exceeded():
                greedy_w, greedy_mask = self._local_1swap_improve(greedy_mask, greedy_w)

            if greedy_w > best_weight + 1e-9:
                best_weight = greedy_w
                best_mask = greedy_mask
                self.anytime.append((time.time() - self.start, best_weight))

        duration = time.time() - self.start
        if not self.anytime or abs(self.anytime[-1][0] - duration) > 1e-9:
            self.anytime.append((duration, best_weight))

        return {
            "best_weight": float(best_weight),
            "best_mask": int(best_mask),
            "nodes": self.restarts,
            "time": duration,
            "anytime": self.anytime,
            "interrupted": self.interrupted,
        }


def resolve_search_modes(name, allow_none=False):
    """Map user-provided search mode names to concrete solver lists."""
    normalized = (name or "").lower()
    mapping = {
        "bnb": ["bnb"],
        "astar": ["astar"],
        "wastar": ["wastar"],
        "ls": ["ls"],
        "both": ["bnb", "astar"],
        "bothls": ["bnb", "astar", "ls"],
        "all": ["bnb", "astar", "wastar"],
        "allls": ["bnb", "astar", "wastar", "ls"],
        "full": ["bnb", "astar", "wastar", "ls"],
    }
    if allow_none and normalized == "none":
        return []
    if normalized not in mapping:
        raise ValueError(f"Unknown search mode '{name}'")
    return mapping[normalized]


def select_cpsat_targets(comp_ptr, comps, topk):
    if topk is None or topk <= 0:
        return set()
    sizes = []
    for cidx in comps:
        size = int(comp_ptr[cidx + 1] - comp_ptr[cidx])
        sizes.append((size, cidx))
    sizes.sort(reverse=True)
    return {c for _, c in sizes[:topk]}


def parse_wastar_schedule(text):
    default = (1.5, 1.25, 1.0)
    if text is None:
        return default
    parts = [p.strip() for p in str(text).split(",")]
    values = [float(p) for p in parts if p]
    if not values:
        raise ValueError("W-A* schedule must include at least one epsilon value")
    return tuple(values)


def solve_components(
    dataset_id,
    algo,
    comps=None,
    time_limit=None,
    node_limit=None,
    validate=False,
    priority_model=None,
    ls_restarts=16,
    ls_rng_seed=0,
    ls_alpha=0.5,
    wastar_schedule=None,
    cpsat_topk=0,
    cpsat_time_limit=60.0,
    cpsat_workers=4,
):
    data, meta = load_dataset(dataset_id)
    comp_ptr = data["comp_ptr"]
    n_components = comp_ptr.size - 1
    if comps is None:
        comps = list(range(n_components))
    search_modes = resolve_search_modes(algo)
    results = []
    total_bnb_weight = 0.0
    total_bnb_nodes = 0
    wastar_schedule = tuple(wastar_schedule or (1.5, 1.25, 1.0))
    cpsat_targets = select_cpsat_targets(comp_ptr, comps, cpsat_topk)

    for cidx in comps:
        component = ComponentView(data, meta, cidx)
        if priority_model is not None:
            component.attach_priority_model(priority_model)
        print(f"Component {cidx} size={component.n}")
        if validate:
            component.validate_upper_bounds(samples=8, subset_size=min(14, component.n))
            print("  UB validation passed")
        component_result = {"component": cidx, "size": component.n}

        for mode in search_modes:
            if mode == "bnb":
                bnb = BranchAndBoundSolver(component, time_limit=time_limit, node_limit=node_limit).solve()
                component_result["bnb"] = bnb
                print(
                    f"  BnB weight={bnb['best_weight']:.2f} nodes={bnb['nodes']} "
                    f"time={bnb['time']:.2f}s interrupted={bnb['interrupted']}"
                )
                total_bnb_weight += bnb["best_weight"]
                total_bnb_nodes += bnb["nodes"]
            elif mode == "astar":
                astar = AStarSolver(component, time_limit=time_limit, node_limit=node_limit).solve()
                component_result["astar"] = astar
                print(
                    f"  A* weight={astar['best_weight']:.2f} nodes={astar['nodes']} "
                    f"time={astar['time']:.2f}s interrupted={astar['interrupted']}"
                )
            elif mode == "wastar":
                wastar = WeightedBestFirstSolver(
                    component, eps_schedule=wastar_schedule, time_limit=time_limit, node_limit=node_limit
                ).solve()
                component_result["wastar"] = wastar
                print(
                    f"  W-A* weight={wastar['best_weight']:.2f} nodes={wastar['nodes']} "
                    f"time={wastar['time']:.2f}s interrupted={wastar['interrupted']}"
                )
            elif mode == "ls":
                ls = MultiStartLocalSearchSolver(
                    component,
                    restarts=ls_restarts,
                    time_limit=time_limit,
                    rng_seed=ls_rng_seed,
                    alpha=ls_alpha,
                ).solve()
                component_result["ls"] = ls
                print(
                    f"  LS weight={ls['best_weight']:.2f} restarts={ls['nodes']} "
                    f"time={ls['time']:.2f}s interrupted={ls['interrupted']}"
                )

        if cidx in cpsat_targets:
            try:
                cpsat = solve_with_cpsat(
                    component, time_limit=cpsat_time_limit, workers=cpsat_workers, scale=1000.0
                )
                component_result["cpsat"] = cpsat
                print(
                    f"  CP-SAT weight={cpsat['best_weight']:.2f} status={cpsat['status']} "
                    f"time={cpsat['time']:.2f}s"
                )
            except RuntimeError as exc:
                component_result["cpsat_error"] = str(exc)
                print(f"  CP-SAT skipped: {exc}")

        results.append(component_result)

    if "bnb" in search_modes:
        print(f"Total weight across components (BnB): {total_bnb_weight:.2f}, total nodes: {total_bnb_nodes}")
    return results


def solve_with_cpsat(component, time_limit=60.0, workers=4, scale=1000.0):
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("CP-SAT backend requires ortools (pip install ortools).") from exc

    model = cp_model.CpModel()
    vars_ = [model.NewBoolVar(f"x_{i}") for i in range(component.n)]
    global_vertices = component.global_vertices
    indptr = component.indptr
    indices = component.indices
    local_index = component.local_index
    for i, gv in enumerate(global_vertices):
        start = indptr[gv]
        end = indptr[gv + 1]
        for nb in indices[start:end]:
            j = local_index.get(int(nb))
            if j is not None and j > i:
                model.Add(vars_[i] + vars_[j] <= 1)

    scaled_terms = []
    for i in range(component.n):
        coeff = int(round(scale * float(component.weights[i])))
        scaled_terms.append(coeff * vars_[i])
    model.Maximize(sum(scaled_terms))

    solver = cp_model.CpSolver()
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = max(1, int(workers or 1))
    status = solver.Solve(model)

    best_mask = 0
    best_weight = 0.0
    for i in range(component.n):
        if solver.Value(vars_[i]):
            best_mask |= 1 << i
            best_weight += float(component.weights[i])

    status_name = solver.StatusName(status) if hasattr(solver, "StatusName") else str(status)
    best_bound = solver.BestObjectiveBound() / scale if math.isfinite(solver.BestObjectiveBound()) else None
    wall = solver.WallTime()
    return {
        "best_weight": best_weight,
        "best_mask": best_mask,
        "status": status_name,
        "time": float(wall),
        "best_bound": best_bound,
        "num_branches": solver.NumBranches(),
        "num_conflicts": solver.NumConflicts(),
    }


def run_greedy_suite(component):
    results = {}
    for name, order in component.greedy_orders.items():
        start = time.time()
        weight, selection = component.greedy_from_order(order)
        elapsed = time.time() - start
        picks = selection.bit_count() if hasattr(int, "bit_count") else bin(selection).count("1")
        results[name] = {"weight": float(weight), "time": elapsed, "picks": picks}
    return results


def run_subgraph_checks(component, samples, subset_size, seed):
    rng = random.Random(seed)
    outputs = []
    for i in range(samples):
        if component.n <= subset_size:
            subset = list(range(component.n))
        else:
            subset = rng.sample(range(component.n), subset_size)
        mask = 0
        for v in subset:
            mask |= 1 << v
        ub = component.upper_bound(mask)
        exact = component.exact_bruteforce_weight(subset)
        outputs.append({"subset_index": i, "size": len(subset), "ub": float(ub), "exact": float(exact)})
    return outputs


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


GREEDY_PLOT_KEYS = ["Density", "Heaviest"]


def _plot_series(ax, label, times, weights, style="step"):
    if not times:
        return False
    if len(times) == 1:
        ax.plot(times, weights, marker="o", linestyle="none", label=label)
        return True
    if style == "step":
        ax.step(times, weights, where="post", label=label)
    else:
        ax.plot(times, weights, linestyle=style, label=label)
    return True


def maybe_plot_anytime(component_idx, comp_result, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots()
    plotted = False
    if "bnb" in comp_result:
        if comp_result["bnb"]["anytime"]:
            t, w = zip(*comp_result["bnb"]["anytime"])
            plotted |= _plot_series(ax, "BnB", t, w)
    greedy = comp_result.get("greedy", {})
    last_time = 0.0
    if comp_result.get("bnb", {}).get("anytime"):
        last_time = max(last_time, comp_result["bnb"]["anytime"][-1][0])
    if comp_result.get("astar", {}).get("anytime"):
        last_time = max(last_time, comp_result["astar"]["anytime"][-1][0])
    if comp_result.get("wastar", {}).get("anytime"):
        last_time = max(last_time, comp_result["wastar"]["anytime"][-1][0])
    last_time = max(last_time, 1e-9)
    for key in GREEDY_PLOT_KEYS:
        data = greedy.get(key.lower())
        if data:
            ax.hlines(
                y=data["weight"],
                xmin=0.0,
                xmax=last_time,
                colors="gray",
                linestyles="dashed",
                label=f"Greedy-{key}",
            )
            plotted = True
    if "astar" in comp_result:
        if comp_result["astar"]["anytime"]:
            t, w = zip(*comp_result["astar"]["anytime"])
            plotted |= _plot_series(ax, "A*", t, w)
    if "wastar" in comp_result:
        if comp_result["wastar"]["anytime"]:
            t, w = zip(*comp_result["wastar"]["anytime"])
            plotted |= _plot_series(ax, "W-A*", t, w)
    if "ls" in comp_result:
        if comp_result["ls"].get("anytime"):
            t, w = zip(*comp_result["ls"]["anytime"])
            plotted |= _plot_series(ax, "LocalSearch", t, w)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Best weight so far")
    ax.set_title(f"Component {component_idx} anytime curves")
    ax.legend()
    ensure_dir(out_dir)
    fig.savefig(out_dir / f"component_{component_idx}_anytime.png", bbox_inches="tight")
    plt.close(fig)


def sanitize_search_result(result):
    sanitized = dict(result)
    mask = sanitized.pop("best_mask", None)
    if mask is not None:
        sanitized["best_mask_hex"] = format(mask, "x")
    if "anytime" in sanitized:
        sanitized["anytime"] = [(float(t), float(w)) for (t, w) in sanitized["anytime"]]
    sanitized["best_weight"] = float(sanitized.get("best_weight", 0.0))
    sanitized["time"] = float(sanitized.get("time", 0.0))
    return sanitized


def evaluate_dataset(
    dataset_id,
    comps=None,
    max_components=None,
    search_mode="both",
    sample_subgraphs=0,
    subset_size=14,
    time_limit=None,
    node_limit=None,
    output_dir="results",
    seed=0,
    priority_model=None,
    ls_restarts=16,
    ls_rng_seed=0,
    ls_alpha=0.5,
    wastar_schedule=None,
    cpsat_topk=0,
    cpsat_time_limit=60.0,
    cpsat_workers=4,
):
    data, meta = load_dataset(dataset_id)
    comp_ptr = data["comp_ptr"]
    total_components = comp_ptr.size - 1
    selected = comps if comps is not None else list(range(total_components))
    if max_components is not None:
        selected = selected[:max_components]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) / dataset_id / timestamp
    ensure_dir(out_root)
    rng = random.Random(seed)

    component_results = []
    totals = {}
    search_modes = resolve_search_modes(search_mode, allow_none=True)
    wastar_schedule = tuple(wastar_schedule or (1.5, 1.25, 1.0))
    cpsat_targets = select_cpsat_targets(comp_ptr, selected, cpsat_topk)

    for cidx in selected:
        component = ComponentView(data, meta, cidx)
        if priority_model is not None:
            component.attach_priority_model(priority_model)
        info = {"component": cidx, "size": component.n}
        print(f"[Eval] Component {cidx} size={component.n}")
        info["greedy"] = run_greedy_suite(component)
        if sample_subgraphs:
            info["subgraph_checks"] = run_subgraph_checks(
                component, sample_subgraphs, subset_size, rng.randint(0, 10**9)
            )
        for mode in search_modes:
            if mode == "bnb":
                bnb = BranchAndBoundSolver(component, time_limit=time_limit, node_limit=node_limit).solve()
                totals["bnb"] = totals.get("bnb", 0.0) + bnb["best_weight"]
                info["bnb"] = sanitize_search_result(bnb)
                print(f"  BnB weight={bnb['best_weight']:.2f} nodes={bnb['nodes']} time={bnb['time']:.2f}s")
            elif mode == "astar":
                astar = AStarSolver(component, time_limit=time_limit, node_limit=node_limit).solve()
                totals["astar"] = totals.get("astar", 0.0) + astar["best_weight"]
                info["astar"] = sanitize_search_result(astar)
                print(f"  A* weight={astar['best_weight']:.2f} nodes={astar['nodes']} time={astar['time']:.2f}s")
            elif mode == "wastar":
                wastar = WeightedBestFirstSolver(
                    component, eps_schedule=wastar_schedule, time_limit=time_limit, node_limit=node_limit
                ).solve()
                totals["wastar"] = totals.get("wastar", 0.0) + wastar["best_weight"]
                info["wastar"] = sanitize_search_result(wastar)
                print(f"  W-A* weight={wastar['best_weight']:.2f} nodes={wastar['nodes']} time={wastar['time']:.2f}s")
            elif mode == "ls":
                ls = MultiStartLocalSearchSolver(
                    component,
                    restarts=ls_restarts,
                    time_limit=time_limit,
                    rng_seed=ls_rng_seed,
                    alpha=ls_alpha,
                ).solve()
                totals["ls"] = totals.get("ls", 0.0) + ls["best_weight"]
                info["ls"] = sanitize_search_result(ls)
                print(f"  LS weight={ls['best_weight']:.2f} restarts={ls['nodes']} time={ls['time']:.2f}s")

        if cidx in cpsat_targets:
            try:
                cpsat = solve_with_cpsat(component, time_limit=cpsat_time_limit, workers=cpsat_workers, scale=1000.0)
                info["cpsat"] = sanitize_search_result(cpsat)
                print(f"  CP-SAT weight={cpsat['best_weight']:.2f} status={cpsat['status']} time={cpsat['time']:.2f}s")
            except RuntimeError as exc:
                info["cpsat_error"] = str(exc)
                print(f"  CP-SAT skipped: {exc}")

        maybe_plot_anytime(cidx, info, out_root / "plots")
        component_results.append(info)

    summary = {
        "dataset": dataset_id,
        "components": component_results,
        "totals": totals,
        "search_mode": search_mode,
        "time_limit": time_limit,
        "node_limit": node_limit,
        "sample_subgraphs": sample_subgraphs,
        "subset_size": subset_size,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Eval] Results saved to {out_root}")
    return summary


def build_priority_model(name, args=None):
    if not name or name == "none":
        return None
    if name == "heuristic":
        from neural_priority import HeuristicPriorityModel

        return HeuristicPriorityModel()
    if name == "degreeaware":
        from neural_priority import DegreeAwarePriorityModel

        alpha = getattr(args, "degree_alpha", 0.5)
        return DegreeAwarePriorityModel(alpha=alpha)
    if name == "gibbs":
        from gibbs_priority import GibbsPriorityModel

        beta = getattr(args, "gibbs_beta", 0.5)
        sweeps = getattr(args, "gibbs_sweeps", 30)
        burn = getattr(args, "gibbs_burn", 10)
        rng_seed = getattr(args, "gibbs_rng_seed", 0)
        return GibbsPriorityModel(beta=beta, sweeps=sweeps, burn=burn, seed=rng_seed)
    raise ValueError(f"Unknown priority model '{name}'")


def main():
    parser = argparse.ArgumentParser(description="Solve/evaluate MIS on frozen satellite datasets.")
    parser.add_argument(
        "--dataset",
        default="worldcities_1d_4sats",
        help="Dataset ID under ./frozen (e.g., worldcities_1d_4sats)",
    )
    parser.add_argument("--mode", choices=["solve", "eval"], default="solve")
    parser.add_argument(
        "--algo",
        choices=["bnb", "astar", "both", "wastar", "all", "ls", "bothls", "allls", "full"],
        default="bnb",
        help=(
            "Solve mode searchers: both=BnB+A*, all=BnB+A*+W-A*, ls=local search, "
            "bothls=BnB+A*+LS, allls/full=BnB+A*+W-A*+LS"
        ),
    )
    parser.add_argument("--components", nargs="+", type=int, help="Specific component indices to target")
    parser.add_argument("--time-limit", type=float, default=None, help="Per-component time cap (seconds)")
    parser.add_argument("--node-limit", type=int, default=None, help="Node expansion cap per component")
    parser.add_argument("--validate-ub", action="store_true", help="Validate upper bounds (solve mode)")
    parser.add_argument("--eval-max-components", type=int, help="Eval mode: maximum number of components")
    parser.add_argument(
        "--eval-search",
        choices=["none", "bnb", "astar", "both", "wastar", "all", "ls", "bothls", "allls", "full"],
        default="both",
        help=(
            "Eval search traces: both=BnB+A*, all=BnB+A*+W-A*, ls=local search, "
            "bothls=BnB+A*+LS, allls/full=BnB+A*+W-A*+LS, none=skip exact search"
        ),
    )
    parser.add_argument("--eval-subgraphs", type=int, default=0, help="Eval mode: random subgraph checks per component")
    parser.add_argument("--eval-subset-size", type=int, default=14, help="Eval mode: induced subgraph size")
    parser.add_argument("--eval-outdir", default="results", help="Eval mode output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for eval mode")
    parser.add_argument(
        "--priority-model",
        choices=["none", "heuristic", "gibbs", "degreeaware"],
        default="none",
        help="Optional vertex priority scorer used for bounds/branching",
    )
    parser.add_argument(
        "--degree-alpha",
        type=float,
        default=0.5,
        help="Penalty for high-degree vertices in DegreeAwarePriorityModel",
    )
    parser.add_argument(
        "--ls-restarts",
        type=int,
        default=16,
        help="Number of randomized greedy restarts for the local-search solver",
    )
    parser.add_argument(
        "--ls-rng-seed",
        type=int,
        default=0,
        help="Base seed for the local-search solver (offset per component)",
    )
    parser.add_argument(
        "--wastar-schedule",
        default="1.5,1.25,1.0",
        help="Comma-separated eps schedule for weighted A* (e.g., 2.0,1.5,1.0)",
    )
    parser.add_argument(
        "--cpsat-topk",
        type=int,
        default=0,
        help="Run CP-SAT on the largest K components (0 disables CP-SAT)",
    )
    parser.add_argument("--cpsat-time-limit", type=float, default=60.0, help="CP-SAT per-component time limit (s)")
    parser.add_argument("--cpsat-workers", type=int, default=4, help="CP-SAT parallel workers")
    parser.add_argument("--gibbs-sweeps", type=int, default=30, help="Gibbs sweeps (if --priority-model gibbs)")
    parser.add_argument("--gibbs-burn", type=int, default=10, help="Gibbs burn-in sweeps")
    parser.add_argument("--gibbs-beta", type=float, default=0.5, help="Gibbs beta (higher biases toward weight)")
    parser.add_argument("--gibbs-rng-seed", type=int, default=0, help="RNG seed for Gibbs priority sampler")
    args = parser.parse_args()
    wastar_schedule = parse_wastar_schedule(args.wastar_schedule)
    priority_model = build_priority_model(args.priority_model, args)

    if args.mode == "solve":
        solve_components(
            dataset_id=args.dataset,
            algo=args.algo,
            comps=args.components,
            time_limit=args.time_limit,
            node_limit=args.node_limit,
            validate=args.validate_ub,
            priority_model=priority_model,
            ls_restarts=args.ls_restarts,
            ls_rng_seed=args.ls_rng_seed,
            ls_alpha=args.degree_alpha,
            wastar_schedule=wastar_schedule,
            cpsat_topk=args.cpsat_topk,
            cpsat_time_limit=args.cpsat_time_limit,
            cpsat_workers=args.cpsat_workers,
        )
    else:
        evaluate_dataset(
            dataset_id=args.dataset,
            comps=args.components,
            max_components=args.eval_max_components,
            search_mode=args.eval_search,
            sample_subgraphs=args.eval_subgraphs,
            subset_size=args.eval_subset_size,
            time_limit=args.time_limit,
            node_limit=args.node_limit,
            output_dir=args.eval_outdir,
            seed=args.seed,
            priority_model=priority_model,
            ls_restarts=args.ls_restarts,
            ls_rng_seed=args.ls_rng_seed,
            ls_alpha=args.degree_alpha,
            wastar_schedule=wastar_schedule,
            cpsat_topk=args.cpsat_topk,
            cpsat_time_limit=args.cpsat_time_limit,
            cpsat_workers=args.cpsat_workers,
        )


if __name__ == "__main__":
    main()
