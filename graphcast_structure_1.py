# Graphcast structure 1 functions #
"""
Latent structure for GraphCast-style models with an icosahedral mesh.

Every processor step shares one latent space of ``latent_dim`` features defined
on the same mesh nodes, so a channel index means the same thing at each step
and optional translator matrices can map earlier steps into the final step's
basis.

Expected on-disk layout (one file per processor step, per month)::

    latent_mesh_step_<step>_<year>_<month>.npz

each holding an array of shape (timesteps, nodes, batch, latent_dim).
"""
import streamlit as st
import re
import os
import pandas as pd
import numpy as np
from utils import apply_translator, mesh_features_to_latlon
from latent_structures import LatentStructure, NodeGeometry, register_structure

def get_available_months_gc_1(model_cfg):
    latent_dir = model_cfg["latent"]["dir"]
    latent_label = model_cfg["latent"]["label"]   # e.g. "flt"

    pattern = re.compile(rf"latent_mesh_step_(\d+)_(\d{{4}})_(\d{{2}})\.npz")

    months = []
    for f in os.listdir(latent_dir):
        m = pattern.match(f)
        if m:
            _, year, month = m.groups()
            months.append((int(year), int(month)))

    return sorted(set(months))

def _timesteps_in_month(year, month, model_cfg):
    """Number of timesteps stored in this month's latent files, or None."""
    latent_dir = model_cfg["latent"]["dir"]
    pattern = re.compile(rf"latent_mesh_step_(\d+)_(\d{{4}})_(\d{{2}})\.npz")
    for f in sorted(os.listdir(latent_dir)):
        m = pattern.match(f)
        if m and int(m.group(2)) == year and int(m.group(3)) == month:
            with np.load(os.path.join(latent_dir, f), allow_pickle=False) as arr:
                return int(arr[arr.files[0]].shape[0])
    return None


def get_available_times_gc_1(year, month, model_cfg):
    if year is None or month is None:
        raise ValueError(
            f"Invalid year/month passed to get_available_times_gc_1: "
            f"year={year}, month={month}"
        )

    start_time = pd.Timestamp(year, month, 1, 12, 0, tz="UTC")

    if month == 12:
        end_time = pd.Timestamp(year + 1, 1, 1, 0, 0, tz="UTC")
    else:
        end_time = pd.Timestamp(year, month + 1, 1, 0, 0, tz="UTC")

    times = pd.date_range(start=start_time, end=end_time, freq="6h", inclusive="left")

    # A latent file may cover only part of the month (demo data, or an
    # interrupted extraction). Offering times the files do not contain would
    # fail later, so trim the axis to what is actually stored.
    n_times = _timesteps_in_month(year, month, model_cfg)
    if n_times is not None:
        times = times[:n_times]
    return times

def load_latent_gc_1(flt_time, model_cfg, batch_num=0, use_translator=False):
    latent_dir = model_cfg["latent"]["dir"]
    translator_dir = model_cfg["latent"]["translator_dir"]

    year = flt_time.year
    month = flt_time.month

    # --- compute timestep index ---
    start_time = pd.Timestamp(year, month, 1, 12, 0, tz="UTC")
    time_index = int((flt_time - start_time) / pd.Timedelta(hours=6))

    pattern = re.compile(rf"latent_mesh_step_(\d+)_(\d{{4}})_(\d{{2}})\.npz")

    parsed = []
    for f in os.listdir(latent_dir):
        m = pattern.match(f)
        if m:
            step, y, mo = m.groups()
            if int(y) == year and int(mo) == month:
                parsed.append({"filename": f, "step": int(step)})

    parsed = sorted(parsed, key=lambda x: x["step"])
    if not parsed:
        raise FileNotFoundError(f"No latent files found for {year}-{month:02d}")

    latent_list = []
    steps = []

    # Translators map an earlier processor step into the final step's basis, so
    # the final step itself is never translated.
    final_step = max(p["step"] for p in parsed)
    translator_template = model_cfg.get(
        "translator_template", "translator_matrix_{step}_gnn.npz"
    )

    for p in parsed:
        filename = os.path.join(latent_dir, p["filename"])
        arr = np.load(filename)
        key = arr.files[0]
        full_latent = arr[key]   # (timesteps, nodes, batch, latent_dim)

        latent_t = full_latent[time_index, :, batch_num, :]

        if use_translator and p["step"] != final_step:
            translator_file = os.path.join(
                translator_dir,
                translator_template.format(step=p["step"]),
            )
            if not os.path.exists(translator_file):
                raise FileNotFoundError(
                    f"Translator matrix not found for processor step {p['step']}: "
                    f"{translator_file}. Turn off 'Use translator', or check "
                    "translator_dir in paths.json."
                )
            latent_t = apply_translator(latent_t, translator_file)

        latent_list.append(latent_t)
        steps.append(p["step"])

    latent_out = np.stack(latent_list, axis=0)  # (processor steps, nodes, latent dimension)
    return latent_out, steps, time_index


def node_geometry_gc_1(model_cfg):
    """Latitude/longitude of the GraphCast mesh nodes.

    The mesh node file stores (cos(theta), cos(phi), sin(phi)) per node; these
    are converted back to degrees. The mesh is unstructured, so no grid shape
    is reported and the app draws latent fields as a scatter of nodes.
    """
    lat, lon = mesh_features_to_latlon(model_cfg["graph_coords_filepath"])
    return NodeGeometry(lat=lat, lon=lon, grid_shape=None)


register_structure(
    LatentStructure(
        name="graphcast_structure_1",
        available_months=get_available_months_gc_1,
        available_times=get_available_times_gc_1,
        load_latent=load_latent_gc_1,
        node_geometry=node_geometry_gc_1,
        step_name="Processor step",
        supports_translator=True,
        ragged_channels=False,
        description=(
            "GraphCast mesh latents: one vector per icosahedral mesh node at "
            "each processor step, all sharing one latent space."
        ),
    )
)
