# ACE / SFNO structure 1 functions #
"""
Latent structure for Ai2 Climate Emulator (ACE) style models.

ACE models (see https://github.com/mahf708/ace, the ``fme`` package) are built
on Spherical Fourier Neural Operators. Unlike GraphCast, their latent state is
not an unstructured mesh: every block of the network holds an embedding of
shape ``(embed_dim, n_lat, n_lon)`` on the model's own latitude-longitude grid
(ACE2 at 1 degree is 180 x 360). Every block shares the same ``embed_dim``, so
a channel index means the same thing at each block and can be compared across
blocks, exactly as GraphCast processor steps can.

This module maps that gridded state onto the app's canonical
``(n_steps, n_nodes, latent_dim)`` form by flattening the grid in row-major
order, so the rest of the app needs no changes.

Expected on-disk layout (one file per block, per month)::

    latent_grid_block_<block>_<year>_<month>.npz

Each file holds one array with dimensions described by
``MODELS[...]["latent"]["dims"]``, by default
``time,batch,channel,lat,lon``. A single file holding every block at once is
also supported: drop ``{step}`` from the filename template and include a
``step`` dimension in ``dims``.

``.nc`` files are read through xarray, in which case ``dims`` may be omitted
and the dimension names of the data variable are used instead.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from latent_structures import (
    LatentStructure,
    grid_latent_to_nodes,
    grid_node_geometry,
    load_coords_file,
    parse_dims,
    parse_month_files,
    read_grid_from_file,
    read_latent_file,
    regular_global_grid,
    register_structure,
    select_time_batch,
    stack_ragged_steps,
    time_index_for,
    times_for_month,
)
from utils import apply_translator

DEFAULT_TEMPLATE = "latent_grid_block_{step}_{year}_{month}.npz"


def _template(model_cfg):
    return model_cfg["latent"].get("filename_template", DEFAULT_TEMPLATE)


def _month_files(model_cfg, year=None, month=None):
    files = parse_month_files(model_cfg["latent"]["dir"], _template(model_cfg))
    if year is not None:
        files = [f for f in files if f["year"] == year and f["month"] == month]
    return files


# ----------------------------------------------------
# --- Structure interface
# ----------------------------------------------------


def get_available_months_ace_1(model_cfg):
    """Sorted (year, month) pairs with latent files on disk."""
    return sorted({(f["year"], f["month"]) for f in _month_files(model_cfg)})


def get_available_times_ace_1(year, month, model_cfg):
    """Time axis of the latents for one month.

    If the latent file stores a ``time`` array that is used directly, so
    irregular or non-standard calendars still work. Otherwise the axis is
    generated from ``timestep_start_hour`` and ``timestep_hours`` /
    ``timestep_freq`` in the model config, truncated to the number of
    timesteps actually present in the file.
    """
    files = _month_files(model_cfg, year, month)
    if not files:
        raise FileNotFoundError(f"No latent files found for {year}-{month:02d}")

    arr, dims, times = read_latent_file(files[0]["path"], model_cfg["latent"])
    if times is not None:
        return pd.DatetimeIndex(times)

    names = parse_dims(dims)
    n_times = arr.shape[names.index("time")] if "time" in names else 1
    return times_for_month(year, month, model_cfg, n_times=n_times)


def load_latent_ace_1(flt_time, model_cfg, batch_num=0, use_translator=False):
    """Load every block's latent grid for one forecast time.

    Returns:
        (latent, steps, time_index) where ``latent`` has shape
        (n_blocks, n_nodes, embed_dim), ``steps`` lists the block indices and
        ``time_index`` is the position of ``flt_time`` in the file.
    """
    flt_time = pd.Timestamp(flt_time)
    year, month = flt_time.year, flt_time.month

    files = _month_files(model_cfg, year, month)
    if not files:
        raise FileNotFoundError(f"No latent files found for {year}-{month:02d}")

    times = get_available_times_ace_1(year, month, model_cfg)
    time_index = time_index_for(flt_time, model_cfg, times=times)

    translator_dir = model_cfg["latent"].get("translator_dir")
    translator_template = model_cfg.get(
        "translator_template", "translator_matrix_{step}_sfno.npz"
    )
    # As for GraphCast, a translator maps an earlier block into the final
    # block's basis, so the final block is never translated.
    final_step = max(f["step"] for f in files)

    blocks: list[np.ndarray] = []
    steps: list[int] = []

    for file_info in files:
        arr, dims, _ = read_latent_file(file_info["path"], model_cfg["latent"])
        block, order = select_time_batch(arr, dims, time_index, batch_num)

        if "step" in order:
            # One file holding every block: split the step axis out.
            step_axis = order.index("step")
            block = np.moveaxis(block, step_axis, 0)
            inner_order = [d for d in order if d != "step"]
            sub_blocks = [
                grid_latent_to_nodes(block[i], inner_order) for i in range(block.shape[0])
            ]
            sub_steps = list(range(block.shape[0]))
        else:
            sub_blocks = [grid_latent_to_nodes(block, order)]
            sub_steps = [file_info["step"]]

        for step, nodes in zip(sub_steps, sub_blocks, strict=True):
            if use_translator and translator_dir and step != final_step:
                translator_file = os.path.join(
                    translator_dir, translator_template.format(step=step)
                )
                if os.path.exists(translator_file):
                    nodes = apply_translator(nodes, translator_file)
            blocks.append(np.asarray(nodes, dtype=np.float32))
            steps.append(step)

    order_idx = np.argsort(steps, kind="stable")
    blocks = [blocks[i] for i in order_idx]
    steps = [steps[i] for i in order_idx]

    latent_out = stack_ragged_steps(blocks)  # (blocks, nodes, embed_dim)
    return latent_out, steps, time_index


@st.cache_data(show_spinner=False)
def node_geometry_ace_1(model_cfg):
    """Latitude/longitude of every grid cell, flattened to nodes.

    Coordinates come from ``grid_coords_filepath`` when the model config sets
    it, otherwise from lat/lon arrays stored inside the latent files, and
    otherwise from a regular global grid inferred from the latent grid shape.
    """
    coords_path = model_cfg.get("grid_coords_filepath") or model_cfg.get(
        "graph_coords_filepath"
    )
    if coords_path:
        lat, lon, valid = load_coords_file(
            coords_path, mask_key=model_cfg.get("mask_key")
        )
        return grid_node_geometry(lat, lon, valid)

    files = _month_files(model_cfg)
    if not files:
        raise FileNotFoundError(
            "No latent files found, so the model grid cannot be determined. "
            "Check latent_dir in paths.json."
        )
    lat, lon = read_grid_from_file(files[0]["path"])
    if lat is not None and lon is not None:
        return grid_node_geometry(lat, lon)

    # Last resort: assume a regular global grid of the latent array's shape.
    arr, dims, _ = read_latent_file(files[0]["path"], model_cfg["latent"])
    names = parse_dims(dims)
    if "lat" not in names or "lon" not in names:
        raise ValueError(
            "Cannot determine the model grid: set grid_coords_filepath in "
            "paths.json, or store lat/lon arrays in the latent files."
        )
    n_lat = arr.shape[names.index("lat")]
    n_lon = arr.shape[names.index("lon")]
    return grid_node_geometry(*regular_global_grid(n_lat, n_lon))


register_structure(
    LatentStructure(
        name="ace_structure_1",
        available_months=get_available_months_ace_1,
        available_times=get_available_times_ace_1,
        load_latent=load_latent_ace_1,
        node_geometry=node_geometry_ace_1,
        step_name="Block",
        supports_translator=True,
        ragged_channels=False,
        description=(
            "Gridded SFNO/ACE latents: one embedding of shape "
            "(embed_dim, lat, lon) per network block."
        ),
    )
)
