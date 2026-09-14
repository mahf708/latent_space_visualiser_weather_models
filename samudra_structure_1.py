# Samudra structure 1 functions #
"""
Latent structure for Samudra-style ocean emulators.

Samudra (M2LInES, wired into ``fme`` as the ``Samudra`` module, see
https://github.com/mahf708/ace) is a ConvNeXt U-Net rather than a stack of
identical processor blocks. Two things follow from that, and this module
handles both so the rest of the app does not have to:

1. **Every level lives on its own grid.** The encoder halves the horizontal
   resolution at each level and the decoder puts it back. Each level is
   resampled (nearest neighbour) onto the model's native output grid so that
   all levels share one set of nodes and can be shown on the same map and
   selected with the same region.

2. **Every level has its own channel width** (the default Samudra
   configuration is ``ch_width = [200, 250, 300, 400]``). Levels are NaN-padded
   to the widest level, and the app treats a non-finite channel as absent at
   that step. Channel *k* at level 1 and channel *k* at level 2 are different
   features, so the structure is marked ``ragged_channels`` and the app warns
   against reading the cross-level line plot as one feature's trajectory.

Land points are set to NaN and excluded from region selection via the wet mask,
so ocean analyses are not contaminated by land values.

Expected on-disk layout (one file per level, per month)::

    latent_level_<level>_<year>_<month>.npz

with dimensions described by ``MODELS[...]["latent"]["dims"]``, by default
``time,batch,channel,lat,lon``. Ocean datasets are usually monthly, so set
``timestep_freq: "MS"`` in the model config.
"""

from __future__ import annotations

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
    resample_grid_to_shape,
    select_time_batch,
    stack_ragged_steps,
    time_index_for,
    times_for_month,
)

DEFAULT_TEMPLATE = "latent_level_{step}_{year}_{month}.npz"


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


def get_available_months_samudra_1(model_cfg):
    """Sorted (year, month) pairs with latent files on disk."""
    return sorted({(f["year"], f["month"]) for f in _month_files(model_cfg)})


def get_available_times_samudra_1(year, month, model_cfg):
    """Time axis of the latents for one month.

    Ocean emulator output is typically monthly, in which case each file holds a
    single timestep. A ``time`` array stored in the file wins; otherwise the
    axis is generated from ``timestep_freq`` / ``timestep_hours``.
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


def load_latent_samudra_1(flt_time, model_cfg, batch_num=0, use_translator=False):
    """Load every U-Net level for one time, on a single common grid.

    ``use_translator`` is accepted for interface compatibility and ignored:
    Samudra levels do not share a latent basis, so there is no linear map
    between them to apply.

    Returns:
        (latent, steps, time_index) where ``latent`` has shape
        (n_levels, n_nodes, max_channel_width) with NaN where a level has no
        such channel or the node is land.
    """
    flt_time = pd.Timestamp(flt_time)
    year, month = flt_time.year, flt_time.month

    files = _month_files(model_cfg, year, month)
    if not files:
        raise FileNotFoundError(f"No latent files found for {year}-{month:02d}")

    times = get_available_times_samudra_1(year, month, model_cfg)
    time_index = time_index_for(flt_time, model_cfg, times=times)

    geometry = node_geometry_samudra_1(model_cfg)
    if geometry.grid_shape is None:
        raise ValueError(
            "Samudra latents must be defined on a grid; provide "
            "grid_coords_filepath in paths.json."
        )
    target_shape = geometry.grid_shape

    blocks: list[np.ndarray] = []
    steps: list[int] = []

    for file_info in files:
        arr, dims, _ = read_latent_file(file_info["path"], model_cfg["latent"])
        block, order = select_time_batch(arr, dims, time_index, batch_num)

        if "step" in order:
            step_axis = order.index("step")
            block = np.moveaxis(block, step_axis, 0)
            inner_order = [d for d in order if d != "step"]
            sub_blocks = list(block)
            sub_steps = list(range(block.shape[0]))
        else:
            inner_order = order
            sub_blocks = [block]
            sub_steps = [file_info["step"]]

        for step, level in zip(sub_steps, sub_blocks, strict=True):
            if list(inner_order) != ["channel", "lat", "lon"]:
                raise ValueError(
                    "Samudra latents must be gridded; expected dimensions "
                    f"(channel, lat, lon) after selecting time and batch, got {inner_order}"
                )
            # Encoder levels are coarser than the output grid; put them back on
            # the model's native grid so every level shares the same nodes.
            level = resample_grid_to_shape(level, target_shape)
            blocks.append(
                grid_latent_to_nodes(level, inner_order).astype(np.float32)
            )
            steps.append(step)

    order_idx = np.argsort(steps, kind="stable")
    blocks = [blocks[i] for i in order_idx]
    steps = [steps[i] for i in order_idx]

    latent_out = stack_ragged_steps(blocks)  # (levels, nodes, max_channels)

    # Blank out land so it never enters the statistics or the maps.
    if geometry.valid is not None:
        latent_out[:, ~geometry.valid, :] = np.nan

    return latent_out, steps, time_index


@st.cache_data(show_spinner=False)
def node_geometry_samudra_1(model_cfg):
    """Ocean grid coordinates and wet mask, flattened to nodes.

    The wet mask (``wet``/``mask``/``ocean_mask``/``valid`` in the coordinates
    file, or ``mask_key`` in the model config) marks ocean points. Land nodes
    are excluded from region selection and from every analysis step.
    """
    coords_path = model_cfg.get("grid_coords_filepath")
    if coords_path:
        lat, lon, valid = load_coords_file(
            coords_path, mask_key=model_cfg.get("mask_key")
        )
        return grid_node_geometry(lat, lon, valid)

    files = _month_files(model_cfg)
    if not files:
        raise FileNotFoundError(
            "No latent files found, so the ocean grid cannot be determined. "
            "Check latent_dir in paths.json."
        )
    lat, lon = read_grid_from_file(files[0]["path"])
    if lat is not None and lon is not None:
        return grid_node_geometry(lat, lon)

    # Fall back to the finest latent grid on disk, assumed regular and global.
    shapes = []
    for file_info in files:
        arr, dims, _ = read_latent_file(file_info["path"], model_cfg["latent"])
        names = parse_dims(dims)
        if "lat" not in names or "lon" not in names:
            raise ValueError(
                "Cannot determine the ocean grid: set grid_coords_filepath in "
                "paths.json, or store lat/lon arrays in the latent files."
            )
        shapes.append((arr.shape[names.index("lat")], arr.shape[names.index("lon")]))
    n_lat, n_lon = max(shapes, key=lambda s: s[0] * s[1])
    return grid_node_geometry(*regular_global_grid(n_lat, n_lon))


register_structure(
    LatentStructure(
        name="samudra_structure_1",
        available_months=get_available_months_samudra_1,
        available_times=get_available_times_samudra_1,
        load_latent=load_latent_samudra_1,
        node_geometry=node_geometry_samudra_1,
        step_name="U-Net level",
        supports_translator=False,
        ragged_channels=True,
        description=(
            "Samudra ocean U-Net: one activation block per encoder/decoder "
            "level, each with its own resolution and channel width."
        ),
    )
)
