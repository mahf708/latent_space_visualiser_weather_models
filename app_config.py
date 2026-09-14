# app_config.py
"""
Central configuration: where data lives, which models are available and how
their latents are read.

Adding a model means adding an entry to ``MODEL_DEFINITIONS`` below and a
matching block in ``paths.json``. Adding a whole new *family* of models means
writing a structure module (see ``graphcast_structure_1.py``,
``ace_structure_1.py``, ``samudra_structure_1.py``) that registers itself with
``latent_structures.register_structure``; the dispatch functions here then pick
it up automatically, with no if/elif chain to extend.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from latent_structures import get_structure, structure_for  # noqa: F401

# Importing the structure modules registers them. Add new ones here.
import graphcast_structure_1  # noqa: F401  (registers graphcast_structure_1)
import ace_structure_1  # noqa: F401  (registers ace_structure_1)
import samudra_structure_1  # noqa: F401  (registers samudra_structure_1)

# paths.json is the user's local copy; paths.example.json is the checked-in
# template pointing at the demo data.
CONFIG_PATH = Path("paths.json")
EXAMPLE_CONFIG_PATH = Path("paths.example.json")


def load_paths():
    """Read data locations, preferring the user's paths.json."""
    path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Missing paths configuration. Copy paths.example.json to paths.json "
            "and edit it to point at your data."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


PATHS = load_paths()

APP_DEFAULTS = {
    "latent_dim": 512,
    "batch_num": 0,
    "processor_step_max": 16,
    "figsize": (11, 7),
    "figsize_hist": (7, 5),
    "dpi": 100,
    "col_1_width": 3,
    "col_2_width": 2,
    "default_model": "graphcast_small",
    "default_latent_label": "flt",   # or "fit"
    "default_use_translator": True,
    "default_n_top": 9,
}


# ---------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------
# Each entry is built from the matching block in paths.json. Models whose
# paths block is absent are simply not offered, so a user who only has
# GraphCast data does not see broken ACE or Samudra entries.


def _definitions(paths):
    """Build the model definitions from the configured data paths."""
    definitions = {}

    gc = paths.get("graphcast_small")
    if gc:
        definitions["graphcast_small"] = {
            "label": "GraphCast small",
            "latent": {
                "dir": gc["latent_dir"],
                "translator_dir": gc.get("translator_dir"),
                "structure": "graphcast_structure_1",
                "label": "flt",   # hardcoded here, not user-selected
                "use_translator_default": True,
                "processor_step_max": 16,
            },
            "reference": {
                "filename": "Graphcast_small_processed_input_{ym_str}.nc",
                "basepath": gc["era5_basepath"],
                "label": "ERA5",
            },
            "graph_coords_filepath": gc["graph_coords_filepath"],
            "translator_template": "translator_matrix_{step}_gnn.npz",
            "timestep_start_hour": 12,
            "timestep_hours": 6,
        }

    ace = paths.get("ace2_era5")
    if ace:
        definitions["ace2_era5"] = {
            "label": ace.get("label", "ACE2 (ERA5, SFNO)"),
            "display_year_offset": ace.get("display_year_offset", 0),
            "latent": {
                "dir": ace["latent_dir"],
                "translator_dir": ace.get("translator_dir"),
                "structure": "ace_structure_1",
                "label": "flt",
                "filename_template": ace.get(
                    "filename_template", "latent_grid_block_{step}_{year}_{month}.npz"
                ),
                "dims": ace.get("dims", "time,batch,channel,lat,lon"),
                "use_translator_default": False,
                "processor_step_max": ace.get("processor_step_max", 7),
            },
            "reference": {
                "filename": ace.get(
                    "reference_filename", "ace2_era5_reference_{ym_str}.nc"
                ),
                "basepath": ace["reference_basepath"],
                "label": ace.get("reference_label", "ERA5"),
            },
            "grid_coords_filepath": ace.get("grid_coords_filepath"),
            "translator_template": "translator_matrix_{step}_sfno.npz",
            "timestep_start_hour": ace.get("timestep_start_hour", 0),
            "timestep_hours": ace.get("timestep_hours", 6),
        }

    samudra = paths.get("samudra_ocean")
    if samudra:
        definitions["samudra_ocean"] = {
            "label": samudra.get("label", "Samudra (ocean U-Net)"),
            "display_year_offset": samudra.get("display_year_offset", 0),
            "latent": {
                "dir": samudra["latent_dir"],
                "structure": "samudra_structure_1",
                "label": "flt",
                "filename_template": samudra.get(
                    "filename_template", "latent_level_{step}_{year}_{month}.npz"
                ),
                "dims": samudra.get("dims", "time,batch,channel,lat,lon"),
                "use_translator_default": False,
                "processor_step_max": samudra.get("processor_step_max", 7),
            },
            "reference": {
                "filename": samudra.get(
                    "reference_filename", "samudra_reference_{ym_str}.nc"
                ),
                "basepath": samudra["reference_basepath"],
                "label": samudra.get("reference_label", "Ocean reanalysis"),
            },
            "grid_coords_filepath": samudra.get("grid_coords_filepath"),
            "mask_key": samudra.get("mask_key"),
            "timestep_start_hour": samudra.get("timestep_start_hour", 0),
            # Ocean emulator output is monthly: one latent timestep per file.
            "timestep_freq": samudra.get("timestep_freq", "MS"),
            "timestep_hours": samudra.get("timestep_hours"),
        }

    return definitions


MODELS = _definitions(PATHS)

if not MODELS:
    raise ValueError(
        "No models are configured. paths.json must contain at least one of: "
        "graphcast_small, ace2_era5, samudra_ocean."
    )


# ---------------------------------------------------------
# DISPATCH TO THE REGISTERED LATENT STRUCTURE
# ---------------------------------------------------------


def get_available_months(model_cfg):
    """Months for which this model has latent data on disk."""
    return structure_for(model_cfg).available_months(model_cfg)


def get_available_times(year, month, model_cfg):
    """Timestamps available within one month of latent data."""
    return structure_for(model_cfg).available_times(year, month, model_cfg)


@st.cache_data(show_spinner=False)
def load_latent(flt_time, model_cfg, batch_num, use_translator=False):
    """Latents for one forecast time as (n_steps, n_nodes, latent_dim)."""
    return structure_for(model_cfg).load_latent(
        flt_time, model_cfg, batch_num, use_translator
    )


def get_node_geometry(model_cfg):
    """Coordinates (and validity mask) of the nodes the latents live on."""
    return structure_for(model_cfg).node_geometry(model_cfg)


def get_step_name(model_cfg):
    """What one entry of the latent step axis is called for this model."""
    return structure_for(model_cfg).step_name


def supports_translator(model_cfg):
    """Whether the translator option is meaningful for this model."""
    structure = structure_for(model_cfg)
    return structure.supports_translator and bool(
        model_cfg["latent"].get("translator_dir")
    )


def has_ragged_channels(model_cfg):
    """True when a channel index means different things at different steps."""
    return structure_for(model_cfg).ragged_channels


# ---------------------------------------------------------
# REFERENCE (PHYSICAL) DATA
# ---------------------------------------------------------


def _reference_cfg(model_cfg):
    """The reference-data block, accepting the older ``era5`` key."""
    cfg = model_cfg.get("reference") or model_cfg.get("era5")
    if cfg is None:
        raise KeyError(
            f"Model {model_cfg.get('label')} has no reference data configured; "
            "add a 'reference' block with basepath and filename."
        )
    return cfg


def reference_label(model_cfg):
    """Display name of the reference dataset, e.g. 'ERA5'."""
    return _reference_cfg(model_cfg).get("label", "Reference")


def _reference_path(model_cfg, time):
    """Resolve the reference file holding ``time``."""
    cfg = _reference_cfg(model_cfg)
    basepath = cfg["basepath"]
    filename = cfg.get("filename")
    if not filename:
        # A single file or store (e.g. a zarr) holding the whole record.
        return basepath
    time = pd.Timestamp(time)
    return os.path.join(
        basepath,
        filename.format(
            ym_str=f"{time.year}{time.month:02d}",
            year=f"{time.year:04d}",
            month=f"{time.month:02d}",
        ),
    )


def _open_reference(path, cfg):
    engine = cfg.get("engine")
    if engine is None and str(path).endswith(".zarr"):
        engine = "zarr"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reference data file not found: {path}. Check paths.json."
        )
    ds = xr.open_dataset(path, engine=engine)
    return ds.assign_coords(time=pd.to_datetime(ds.time.values))


@st.cache_data(show_spinner=False)
def load_reference_pair(flt_time, model_cfg):
    """Load the reference field at the forecast time and the step before it.

    The pair is (fit, flt): ``fit`` is the model's initialisation state and
    ``flt`` the latent forecast time, so their difference is the change the
    model predicted over one step.

    ``timestep_hours`` fixes the spacing when the model config sets it
    (GraphCast and ACE both step 6-hourly). When it is null the previous
    timestamp present in the file is used instead, which is what monthly ocean
    data needs.

    Returns:
        xarray.Dataset with exactly two times, ordered [fit, flt].
    """
    cfg = _reference_cfg(model_cfg)

    flt_time = pd.Timestamp(flt_time)
    if flt_time.tz is not None:
        flt_time = flt_time.tz_convert("UTC").tz_localize(None)

    timestep_hours = model_cfg.get("timestep_hours")
    fit_time = (
        flt_time - pd.Timedelta(hours=timestep_hours)
        if timestep_hours
        else None
    )

    # Open the file holding the earlier of the two times, and, if the pair
    # straddles a month boundary, the file holding the later one as well.
    base_path = _reference_path(model_cfg, fit_time or flt_time)
    ds = _open_reference(base_path, cfg)
    flt_path = _reference_path(model_cfg, flt_time)
    if flt_path != base_path:
        ds = xr.concat([ds, _open_reference(flt_path, cfg)], dim="time").sortby("time")

    times = pd.DatetimeIndex(ds.time.values)

    if fit_time is None and (len(times) == 0 or times[0] >= flt_time):
        # With one file per month and no fixed spacing (monthly ocean output),
        # the preceding state lives in the previous month's file.
        previous_path = _reference_path(model_cfg, flt_time - pd.Timedelta(days=1))
        if previous_path != flt_path and os.path.exists(previous_path):
            ds = xr.concat(
                [_open_reference(previous_path, cfg), ds], dim="time"
            ).sortby("time")
            times = pd.DatetimeIndex(ds.time.values)

    if fit_time is None:
        # No fixed spacing: use the timestamp immediately before flt_time.
        position = int(np.searchsorted(times, flt_time))
        if position >= len(times) or times[position] != flt_time:
            raise ValueError(
                f"Could not find flt={flt_time} in the reference data for this month."
            )
        if position == 0:
            raise ValueError(
                f"flt={flt_time} is the first timestamp in the reference data, so "
                "there is no preceding state to difference against."
            )
        fit_time = times[position - 1]

    missing = [t for t in (fit_time, flt_time) if t not in times]
    if missing:
        raise ValueError(
            f"Could not find {', '.join(str(t) for t in missing)} in the reference "
            f"data at {flt_path}."
        )

    ds_pair = ds.sel(time=[fit_time, flt_time])

    if ds_pair.sizes["time"] != 2:
        raise ValueError(
            f"Could not find both fit={fit_time} and flt={flt_time} in the reference data."
        )

    return ds_pair


# Kept so existing scripts and notebooks continue to work.
load_era5_fit_and_flt = load_reference_pair
