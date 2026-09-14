#!/usr/bin/env python3
"""Interpretability analyses of SamudrACE-E3SMv3 latents, beyond the app's five steps.

Run after ``run.sh`` has extracted latents and built reference fields::

    demo_data/samudrace_e3smv3/.venv/bin/python examples/samudrace-e3smv3/analysis.py \\
        demo_data/samudrace_e3smv3 analysis_out

It writes map PNGs (drawn with the app's own plotting) and ``results.json``
with every number behind the charts. Latents are read through the app's own
loaders, so these analyses see exactly what the app sees.

Analyses, each for the ocean (Samudra) and the atmosphere (SFNO):

1. probes      - ridge-regression readout of the input state and of the one-step
                 tendency from each level/block, scored on a held-out hemisphere
2. cka         - linear CKA between every pair of levels/blocks, and between the
                 ocean and atmosphere representations of the same initial state
3. dimension   - participation ratio of the channel covariance at each step
4. centring    - cosine similarity with and without removing the global mean vector
5. dictionary  - the channel whose map best tracks a physical variable, scored on
                 anomalies from the zonal mean so latitude alone cannot win
6. regions     - centred cosine similarity to regional mean vectors
7. spread      - latent and physical divergence between two seeds of the
                 stochastic model (needs run_seed.sh SEED; SPREAD_SEED picks it)

Probes and dictionary scores use anomalies from each latitude's mean, so a
readout cannot score well just by knowing that the equator is warm.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import xarray as xr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

# The app modules import streamlit for caching and messages only.
_st = types.ModuleType("streamlit")
_st.cache_data = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
for _name in ("write", "success", "error", "warning", "info", "caption", "markdown"):
    setattr(_st, _name, lambda *a, **k: None)
sys.modules.setdefault("streamlit", _st)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

COMPONENTS = {
    "ocean": {
        "model_key": "samudra_ocean",
        "step_name": "level",
        "probe_targets": {
            "sst": "Sea surface temperature",
            "ssh": "Sea surface height",
            "ocean_sea_ice_fraction": "Sea ice fraction (where defined)",
            "temperatureCoarsened_10": "Temperature, 410-530 m",
        },
        # Sea ice fraction is left out: it is only defined poleward of about 35
        # degrees and is nearly 0 or 1 there, so a correlation says little.
        "dictionary_targets": {
            "sst": "Sea surface temperature",
            "salinityCoarsened_0": "Sea surface salinity",
            "ssh": "Sea surface height",
        },
        "centring_region": ("North Atlantic", 55.0, -35.0, 10.0),
        "centring_step": 0,
        "regions": [
            ("Nino 3.4", 0.0, -145.0, 8.0),
            ("Antarctic Circumpolar Current", -55.0, 60.0, 8.0),
            ("Kuroshio Extension", 37.0, 150.0, 6.0),
        ],
        "region_step": 0,
        "spread_variables": ["sst", "ssh"],
    },
    "atmosphere": {
        "model_key": "ace2_era5",
        "step_name": "block",
        "probe_targets": {
            "PS": "Surface pressure",
            "Tat2m": "2 m temperature",
            "U_2": "Zonal wind, jet level",
            "STW_7": "Total water, lowest level",
        },
        "dictionary_targets": {
            "PS": "Surface pressure",
            "Tat2m": "2 m temperature",
            "U_2": "Zonal wind, jet level",
        },
        "centring_region": ("North Atlantic", 45.0, -30.0, 10.0),
        "centring_step": 7,
        "regions": [
            ("Southern Ocean storm track", -50.0, 90.0, 8.0),
            ("East Pacific ITCZ", 8.0, -120.0, 6.0),
            ("Tibetan Plateau", 32.0, 88.0, 6.0),
        ],
        "region_step": 7,
        "spread_variables": ["Tat2m", "PS"],
    },
}


# ----------------------------------------------------
# --- Loading
# ----------------------------------------------------


class Component:
    """Latents, geometry and reference fields for one model component."""

    def __init__(self, name, data_dir, latent_dir=None):
        from app_config import (
            MODELS,
            get_available_months,
            get_available_times,
            get_node_geometry,
            load_latent,
        )

        self.name = name
        self.spec = COMPONENTS[name]
        self.cfg = dict(MODELS[self.spec["model_key"]])
        if latent_dir is not None:
            self.cfg["latent"] = dict(self.cfg["latent"], dir=latent_dir)
        self._load_latent = load_latent
        year, month = get_available_months(self.cfg)[-1]
        self.times = get_available_times(year, month, self.cfg)
        self.geometry = get_node_geometry(self.cfg)
        self.offset = self.cfg.get("display_year_offset", 0)
        reference_path = os.path.join(
            self.cfg["reference"]["basepath"], self.cfg["reference"]["filename"]
        )
        self.reference = xr.open_dataset(reference_path)
        self._cache = {}

    def latent(self, time_index):
        """(steps, nodes, channels) latents at one captured time."""
        if time_index not in self._cache:
            self._cache.clear()
            latent, _, _ = self._load_latent(self.times[time_index], self.cfg, 0, False)
            self._cache[time_index] = latent
        return self._cache[time_index]

    def model_time(self, time_index):
        """The model date a captured latent is labelled with (its output time)."""
        t = self.times[time_index]
        return f"{t.year - self.offset:04d}-{t.month:02d}-{t.day:02d} {t.hour:02d}:00"

    def reference_time(self, reference_index):
        """The model date of a reference field; index t is the input to latent t."""
        t = pd.Timestamp(self.reference.time.values[reference_index])
        return f"{t.year - self.offset:04d}-{t.month:02d}-{t.day:02d} {t.hour:02d}:00"

    def field(self, variable, reference_index):
        """A reference field flattened to nodes (row-major, like the latents)."""
        values = self.reference[variable].isel(time=reference_index).values
        return np.asarray(values, dtype=np.float64).reshape(-1)

    @property
    def n_steps(self):
        return self.latent(0).shape[0]


def zonal_anomaly(values, lat, mask):
    """Subtract each latitude's mean (over ``mask``) from ``values``."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    for latitude in np.unique(lat):
        row = (lat == latitude) & mask
        if row.any():
            out[row] = values[row] - values[row].mean()
    return out


def step_view(latent, step):
    from utils import step_view as _step_view

    return _step_view(latent, step)


# ----------------------------------------------------
# --- 1. Probes
# ----------------------------------------------------


def ridge_fit(X, Y, lam):
    """Ridge weights for standardised X (n, d) and centred Y (n, k)."""
    n, d = X.shape
    gram = X.T @ X + lam * n * np.eye(d)
    return np.linalg.solve(gram, X.T @ Y)


def r2_score(y, prediction):
    residual = ((y - prediction) ** 2).sum(axis=0)
    total = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - residual / total


def _fold_stats(X, Y, fold, n_folds):
    """Per-fold Gram matrices and cross products, so any union of folds is a sum."""
    stats = []
    for f in range(n_folds):
        m = fold == f
        Xf, Yf = X[m], Y[m]
        stats.append({"n": int(m.sum()), "XX": Xf.T @ Xf, "XY": Xf.T @ Yf,
                      "y": Yf.sum(0), "yy": (Yf ** 2).sum(0)})
    return stats


def _solve(stats, folds, lam, d):
    n = sum(stats[f]["n"] for f in folds)
    XX = sum(stats[f]["XX"] for f in folds)
    XY = sum(stats[f]["XY"] for f in folds)
    return np.linalg.solve(XX + lam * n * np.eye(d), XY)


def _r2(stats, folds, W):
    """R2 on the union of ``folds`` for weights W (d, k), from sufficient statistics."""
    n = sum(stats[f]["n"] for f in folds)
    XX = sum(stats[f]["XX"] for f in folds)
    XY = sum(stats[f]["XY"] for f in folds)
    y = sum(stats[f]["y"] for f in folds)
    yy = sum(stats[f]["yy"] for f in folds)
    sse = yy - 2 * (W * XY).sum(0) + np.einsum("ik,ij,jk->k", W, XX, W)
    sst = yy - y ** 2 / n
    return 1 - sse / sst


def blocked_ridge_r2(X, Y, fold, lambdas, n_folds=5):
    """Mean held-out R2 over folds, with lambda chosen per target by inner CV."""
    d = X.shape[1]
    stats = _fold_stats(X, Y, fold, n_folds)
    outer = []
    for test in range(n_folds):
        train = [f for f in range(n_folds) if f != test]
        inner = np.zeros((len(lambdas), Y.shape[1]))
        for i, lam in enumerate(lambdas):
            for val in train:
                fit = [f for f in train if f != val]
                inner[i] += _r2(stats, [val], _solve(stats, fit, lam, d))
        best = inner.argmax(0)
        scores = np.empty(Y.shape[1])
        for i in set(best.tolist()):
            W = _solve(stats, train, lambdas[i], d)
            r2 = _r2(stats, [test], W)
            scores[best == i] = r2[best == i]
        outer.append(scores)
    return np.mean(outer, axis=0)


def hemisphere_r2(X, Y, lon, lambdas):
    """Fit west of the prime meridian, score east; lambda by inner west split."""
    d = X.shape[1]
    fold = np.where(lon < -90, 0, np.where(lon < 0, 1, 2))
    stats = _fold_stats(X, Y, fold, 3)
    inner = np.zeros((len(lambdas), Y.shape[1]))
    for i, lam in enumerate(lambdas):
        inner[i] += _r2(stats, [1], _solve(stats, [0], lam, d))
        inner[i] += _r2(stats, [0], _solve(stats, [1], lam, d))
    best = inner.argmax(0)
    scores = np.empty(Y.shape[1])
    for i in set(best.tolist()):
        scores[best == i] = _r2(stats, [2], _solve(stats, [0, 1], lambdas[i], d))[best == i]
    return scores


def probes(component, stride=2):
    """Spatially blocked cross-validated R2 of state and tendency anomalies, per step.

    Also reports the harsher west-to-east hemisphere transfer score.
    """
    geometry = component.geometry
    n_lat, n_lon = geometry.grid_shape
    subsample = np.zeros((n_lat, n_lon), dtype=bool)
    subsample[::stride, ::stride] = True
    subsample = subsample.reshape(-1)

    targets = list(component.spec["probe_targets"])
    n_times = len(component.times)
    n_steps = component.n_steps
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    results = {kind: {v: [None] * n_steps for v in targets}
               for kind in ("state", "tendency", "hemisphere_state", "hemisphere_tendency")}

    # Gather the subsampled rows for every step, reading each timestep once.
    rows = {step: {"X": [], "state": [], "tendency": [], "lon": [], "lat": []}
            for step in range(n_steps)}
    channels_by_step = {}
    for t in range(n_times):
        latent = component.latent(t)
        state = np.stack([component.field(v, t) for v in targets], axis=1)
        tendency = np.stack([component.field(v, t + 1) for v in targets], axis=1) - state
        finite = np.isfinite(state).all(1) & np.isfinite(tendency).all(1)
        for step in range(n_steps):
            has_data, channels = step_view(latent, step)
            channels = channels_by_step.setdefault(step, channels)
            use = subsample & has_data & finite
            anomalies = {
                "state": np.stack([zonal_anomaly(state[:, i], geometry.lat, use)
                                   for i in range(len(targets))], 1),
                "tendency": np.stack([zonal_anomaly(tendency[:, i], geometry.lat, use)
                                      for i in range(len(targets))], 1),
            }
            rows[step]["X"].append(latent[step][np.ix_(use, channels)].astype(np.float64))
            rows[step]["state"].append(anomalies["state"][use])
            rows[step]["tendency"].append(anomalies["tendency"][use])
            rows[step]["lon"].append(geometry.lon[use])
            rows[step]["lat"].append(geometry.lat[use])

    for step in range(n_steps):
        X = np.concatenate(rows[step]["X"])
        lon = np.concatenate(rows[step]["lon"])
        lat = np.concatenate(rows[step]["lat"])
        # Spatially blocked cross-validation: 15x15 degree blocks, 5 folds. Blocks
        # are big enough that neighbouring, nearly identical cells cannot sit on
        # both sides of a split, and small enough that every ocean basin appears
        # in every training set.
        block = (np.floor((lat + 90) / 15) * 24 + np.floor((lon + 180) / 15)).astype(int)
        fold_of_block = np.random.default_rng(step).integers(0, 5, size=12 * 24)
        fold = fold_of_block[block]
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        hemisphere = {}
        for kind in ("state", "tendency"):
            Y = np.concatenate(rows[step][kind])
            Y = Y - Y.mean(0)
            scores = blocked_ridge_r2(Xs, Y, fold, lambdas)
            for j, variable in enumerate(targets):
                results[kind][variable][step] = round(float(scores[j]), 4)
            # The harsher test: fit on the western hemisphere, score on the east.
            hemisphere[kind] = hemisphere_r2(Xs, Y, lon, lambdas)
            for j, variable in enumerate(targets):
                results["hemisphere_" + kind][variable][step] = round(float(hemisphere[kind][j]), 4)
        rows[step] = None   # free memory as we go
        print(f"  probes {component.name} step {step}: "
              + " ".join(f"{v}={results['state'][v][step]:.2f}/{results['tendency'][v][step]:.2f}"
                         for v in targets), flush=True)
    return {
        "variables": {v: component.spec["probe_targets"][v] for v in targets},
        "n_samples_per_step": int(X.shape[0]),
        **results,
    }


# ----------------------------------------------------
# --- 2. CKA
# ----------------------------------------------------


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    cross = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(cross / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def representations(component, nodes, time_index=0):
    latent = component.latent(time_index)
    reps = []
    for step in range(latent.shape[0]):
        _, channels = step_view(latent, step)
        reps.append(latent[step][np.ix_(nodes, channels)].astype(np.float64))
    return reps


def cka_matrix(reps_a, reps_b):
    return [[round(linear_cka(a, b), 4) for b in reps_b] for a in reps_a]


# ----------------------------------------------------
# --- 3. Dimensionality
# ----------------------------------------------------


def participation_ratio(component, time_index=0):
    latent = component.latent(time_index)
    out = []
    for step in range(latent.shape[0]):
        has_data, channels = step_view(latent, step)
        X = latent[step][np.ix_(has_data, channels)].astype(np.float64)
        X -= X.mean(0)
        eigenvalues = np.clip(np.linalg.eigvalsh(X.T @ X / len(X)), 0, None)[::-1]
        pr = eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()
        cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
        out.append({
            "step": step,
            "channels": int(len(channels)),
            "participation_ratio": round(float(pr), 2),
            "components_for_90pct": int(np.searchsorted(cumulative, 0.9) + 1),
        })
    return out


# ----------------------------------------------------
# --- 4 & 6. Cosine similarity, centred and not
# ----------------------------------------------------


def region_nodes(component, lat, lon, radius_deg, has_data):
    from utils import select_nodes_within_radius

    g = component.geometry
    return select_nodes_within_radius(g.lat, g.lon, lat, lon, radius_deg * 111,
                                      valid=has_data)


def cosine_map(component, step, reference_vector_nodes, centred, time_index=0):
    from utils import cosine_similarity_to, scatter_to_nodes

    latent = component.latent(time_index)
    has_data, channels = step_view(latent, step)
    X = latent[step][np.ix_(has_data, channels)].astype(np.float64)
    if centred:
        X = X - X.mean(0)
    positions = np.searchsorted(np.flatnonzero(has_data), reference_vector_nodes)
    reference = X[positions].mean(0)
    sims = cosine_similarity_to(reference, X)
    return scatter_to_nodes(sims, has_data, latent.shape[1]), has_data


def save_map(component, values, title, path, cmap, circle=None, vmax=None):
    from utils import make_circle_points, plot_global_overlay_only

    circle_lats = circle_lons = None
    if circle is not None:
        circle_lats, circle_lons = make_circle_points(*circle)
    fig = plot_global_overlay_only(
        title=title,
        figsize=(10, 5),
        overlay_lats=component.geometry.lat,
        overlay_lons=component.geometry.lon,
        overlay_values=values,
        circle_lats=circle_lats,
        circle_lons=circle_lons,
        cmap=cmap,
        grid_shape=component.geometry.grid_shape,
        dpi=90,
    )
    fig.savefig(path, dpi=90, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def save_field_map(component, variable, reference_index, title, path, cmap="viridis"):
    """A physical field on the reference grid, in the app's style."""
    import matplotlib.pyplot as plt

    from utils import plot_global_data_with_overlay

    ds = component.reference
    fig = plot_global_data_with_overlay(
        data=ds[variable].isel(time=reference_index).values,
        lon=ds["lon"].values,
        lat=ds["lat"].values,
        title=title,
        cmap=cmap,
        figsize=(10, 5),
        dpi=90,
    )
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------
# --- 5. Channel dictionary
# ----------------------------------------------------


def dictionary(component, out_dir, time_index=0):
    geometry = component.geometry
    latent = component.latent(time_index)
    entries = []
    for variable, label in component.spec["dictionary_targets"].items():
        field = component.field(variable, time_index)
        best = None
        for step in range(latent.shape[0]):
            has_data, channels = step_view(latent, step)
            use = has_data & np.isfinite(field)
            y_anom = zonal_anomaly(field, geometry.lat, use)[use]
            y_raw = field[use]
            X = latent[step][np.ix_(use, channels)].astype(np.float64)
            X_anom = np.empty_like(X)
            lat_use = geometry.lat[use]
            for latitude in np.unique(lat_use):
                row = lat_use == latitude
                X_anom[row] = X[row] - X[row].mean(0)

            def corr(A, b):
                A = A - A.mean(0)
                b = b - b.mean()
                denominator = np.linalg.norm(A, axis=0) * np.linalg.norm(b)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(denominator > 0, (A.T @ b) / denominator, 0.0)

            r_anom = corr(X_anom, y_anom)
            r_raw = corr(X, y_raw)
            j = int(np.argmax(np.abs(r_anom)))
            if best is None or abs(r_anom[j]) > abs(best["r_anomaly"]):
                from scipy.stats import spearmanr

                # Rank correlation of the same pair: much lower than Pearson means
                # a few extreme cells (an enclosed sea, a plateau) carry the match.
                spearman = spearmanr(X_anom[:, j], y_anom).statistic
                best = {
                    "r_anomaly_spearman": round(float(spearman), 3),
                    "variable": variable,
                    "label": label,
                    "step": step,
                    "channel": int(channels[j]),
                    "r_anomaly": round(float(r_anom[j]), 3),
                    "r_raw": round(float(r_raw[j]), 3),
                    "best_raw_r_any_channel": round(float(np.max(np.abs(r_raw))), 3),
                }
        step, channel = best["step"], best["channel"]
        sign = np.sign(best["r_anomaly"]) or 1.0
        channel_path = f"{component.name}-dict-{variable}-channel.png"
        field_path = f"{component.name}-dict-{variable}-field.png"
        save_map(
            component, sign * latent[step, :, channel],
            f"{component.spec['step_name'].capitalize()} {step}, channel {channel}"
            + (" (sign flipped)" if sign < 0 else ""),
            os.path.join(out_dir, channel_path), "PuOr_r",
        )
        save_field_map(component, variable, time_index,
                       f"{label}, {component.reference_time(time_index)} (input state)",
                       os.path.join(out_dir, field_path),
                       cmap="Blues_r" if "ice" in variable else "viridis")
        best.update(channel_image=channel_path, field_image=field_path,
                    input_time=component.reference_time(time_index))
        entries.append(best)
        print(f"  dictionary {component.name} {variable}: step {step} ch {channel} "
              f"r_anom={best['r_anomaly']} r_raw={best['r_raw']}")
    return entries


# ----------------------------------------------------
# --- 7. Stochastic spread
# ----------------------------------------------------


def spread(component, other):
    """Normalised RMS latent difference between two seeds, per time and step."""
    per_time = []
    for t in range(len(component.times)):
        a_all, b_all = component.latent(t), None
        b_all = other.latent(t)
        row = []
        for step in range(a_all.shape[0]):
            has_data, channels = step_view(a_all, step)
            a = a_all[step][np.ix_(has_data, channels)].astype(np.float64)
            b = b_all[step][np.ix_(has_data, channels)].astype(np.float64)
            scale = np.sqrt(((a - a.mean(0)) ** 2).mean())
            row.append(round(float(np.sqrt(((a - b) ** 2).mean()) / scale), 4))
        per_time.append({"time": component.model_time(t), "normalised_rms": row})
        print(f"  spread {component.name} {component.model_time(t)}: {row}")
    return per_time


def physical_spread(component, run_a, run_b, variables):
    folder = {"ocean": "ocean", "atmosphere": "atmosphere"}[component.name]
    a = xr.open_dataset(os.path.join(run_a, folder, "autoregressive_predictions.nc"),
                        decode_timedelta=False)
    b = xr.open_dataset(os.path.join(run_b, folder, "autoregressive_predictions.nc"),
                        decode_timedelta=False)
    out = {}
    for variable in variables:
        va = a[variable].isel(sample=0).values
        vb = b[variable].isel(sample=0).values
        n = min(len(va), 4 if component.name == "ocean" else len(va))
        diffs = []
        for t in range(n):
            ok = np.isfinite(va[t]) & np.isfinite(vb[t])
            rms = np.sqrt(np.mean((va[t][ok] - vb[t][ok]) ** 2))
            std = np.std(va[t][ok])
            diffs.append(round(float(rms / std), 4))
        out[variable] = diffs
    return out


# ----------------------------------------------------
# --- Entry point
# ----------------------------------------------------


def main():
    data_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "demo_data/samudrace_e3smv3")
    out_dir = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_dir, "analysis_out"))
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(data_dir)

    results_path = os.path.join(out_dir, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    def wanted(name):
        return only is None or name in only

    comps = {name: Component(name, data_dir) for name in COMPONENTS}
    for name, comp in comps.items():
        results.setdefault(name, {})
        results[name]["times"] = [comp.model_time(t) for t in range(len(comp.times))]
        results[name]["n_steps"] = comp.n_steps

    start = time.time()

    if wanted("probes"):
        for name, comp in comps.items():
            results[name]["probes"] = probes(comp)

    if wanted("dimension"):
        for name, comp in comps.items():
            results[name]["dimension"] = participation_ratio(comp)
            print(f"  dimension {name}: {[d['participation_ratio'] for d in results[name]['dimension']]}")

    if wanted("cka"):
        rng = np.random.default_rng(0)
        ocean, atmos = comps["ocean"], comps["atmosphere"]
        ocean_valid = np.flatnonzero(step_view(ocean.latent(0), 0)[0])
        nodes = np.sort(rng.choice(ocean_valid, size=8000, replace=False))
        ocean_reps = representations(ocean, nodes)
        atmos_reps = representations(atmos, nodes)
        results["ocean"]["cka"] = cka_matrix(ocean_reps, ocean_reps)
        results["atmosphere"]["cka"] = cka_matrix(atmos_reps, atmos_reps)
        results["cross_cka"] = {
            "rows": "atmosphere block", "columns": "ocean level",
            "matrix": cka_matrix(atmos_reps, ocean_reps),
            "n_nodes": int(len(nodes)),
            "note": "both captured from the forward pass that reads the same "
                    "initial condition, on the same ocean nodes",
        }
        print("  cka done")

    if wanted("centring"):
        for name, comp in comps.items():
            label, lat, lon, radius = comp.spec["centring_region"]
            step = comp.spec["centring_step"]
            has_data, _ = step_view(comp.latent(0), step)
            nodes = region_nodes(comp, lat, lon, radius, has_data)
            entry = {"region": label, "step": step, "n_nodes": int(len(nodes))}
            for centred in (False, True):
                values, has_data = cosine_map(comp, step, nodes, centred)
                kind = "centred" if centred else "uncentred"
                path = f"{name}-cosine-{kind}.png"
                save_map(comp, values,
                         f"{'Centred' if centred else 'Uncentred'} cosine similarity, "
                         f"{comp.spec['step_name']} {step}",
                         os.path.join(out_dir, path), "PRGn", circle=(lat, lon, radius))
                finite = values[has_data]
                entry[kind] = {
                    "image": path,
                    "median": round(float(np.median(finite)), 3),
                    "fraction_above_0_5": round(float((finite > 0.5).mean()), 3),
                }
            results[name]["centring"] = entry
            print(f"  centring {name}: {entry}")

    if wanted("dictionary"):
        for name, comp in comps.items():
            results[name]["dictionary"] = dictionary(comp, out_dir)

    if wanted("regions"):
        for name, comp in comps.items():
            step = comp.spec["region_step"]
            has_data, _ = step_view(comp.latent(0), step)
            entries = []
            for i, (label, lat, lon, radius) in enumerate(comp.spec["regions"]):
                nodes = region_nodes(comp, lat, lon, radius, has_data)
                values, _ = cosine_map(comp, step, nodes, centred=True)
                path = f"{name}-region-{i}.png"
                save_map(comp, values, f"{label}: centred cosine similarity, "
                         f"{comp.spec['step_name']} {step}",
                         os.path.join(out_dir, path), "PRGn", circle=(lat, lon, radius))
                finite = values[has_data]
                entries.append({"region": label, "lat": lat, "lon": lon, "step": step,
                                "n_nodes": int(len(nodes)), "image": path,
                                "fraction_above_0_5": round(float((finite > 0.5).mean()), 4)})
            results[name]["regions"] = entries
            print(f"  regions {name}: {[(e['region'], e['n_nodes'], e['fraction_above_0_5']) for e in entries]}")

    if wanted("spread"):
        seed = os.environ.get("SPREAD_SEED", "1")   # the seed run_seed.sh used
        seed_dirs = {"ocean": f"latents_seed{seed}/ocean",
                     "atmosphere": f"latents_seed{seed}/atmos"}
        run_dirs = {"ocean": ("run_ocean", f"run_seed{seed}_ocean"),
                    "atmosphere": ("run_atmosphere", f"run_seed{seed}_atmosphere")}
        for name, comp in comps.items():
            if not os.path.isdir(seed_dirs[name]):
                print(f"  spread {name}: {seed_dirs[name]} missing, skipped")
                continue
            other = Component(name, data_dir, latent_dir=seed_dirs[name])
            results[name]["spread"] = {
                "latent": spread(comp, other),
                "physical": physical_spread(comp, *run_dirs[name],
                                            comp.spec["spread_variables"]),
            }
            print(f"  spread {name} physical: {results[name]['spread']['physical']}")

    results["generated_seconds"] = round(time.time() - start, 1)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
