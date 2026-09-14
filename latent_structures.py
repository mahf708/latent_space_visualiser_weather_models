# latent_structures.py
"""
Registry and shared helpers for latent data structures.

A "latent structure" describes how one family of AI weather/climate models
stores its latent representations on disk, and how those latents map onto
geographic coordinates. Registering a structure is all that is needed for the
app to offer a new model family; the app itself stays model agnostic.

The app only ever sees latents in one canonical form:

    latent: np.ndarray of shape (n_steps, n_nodes, latent_dim)

where a "node" is any location with a latitude and longitude. For GraphCast
that is an icosahedral mesh node; for ACE/SFNO or Samudra it is a cell of a
regular latitude-longitude grid that has been flattened in row-major (lat
outer, lon inner) order. Channels that do not exist at a given step (ragged
architectures such as a U-Net, where each level has its own channel width) are
filled with NaN, and the app treats non-finite channels as absent.

Contents:

1 - NodeGeometry - coordinates (and validity mask) of the latent nodes
2 - LatentStructure / register_structure / get_structure - the registry
3 - grid_node_geometry - build a NodeGeometry from 1-D or 2-D lat/lon
4 - load_coords_file - read lat/lon (and an optional mask) from .npy/.npz/.nc
5 - transpose_to_canonical - reorder an on-disk latent array via a dims string
6 - grid_latent_to_nodes - flatten a (channel, lat, lon) block into nodes
7 - resample_grid_to_shape - nearest-neighbour resample of coarse U-Net levels
8 - stack_ragged_steps - NaN-pad steps with differing channel widths
9 - parse_month_files - shared <year>_<month> filename scanning
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Callable, Sequence

import numpy as np

# ----------------------------------------------------
# --- 1. Node geometry
# ----------------------------------------------------


@dataclasses.dataclass
class NodeGeometry:
    """Geographic description of the nodes a latent array is defined on.

    Attributes:
        lat: (n_nodes,) latitudes in degrees, -90 -> 90.
        lon: (n_nodes,) longitudes in degrees, -180 -> 180.
        grid_shape: (n_lat, n_lon) when the nodes are a regular grid flattened
            in row-major order, otherwise None (e.g. an unstructured mesh).
            When set, the app draws fields with pcolormesh instead of a
            scatter, which is both faster and visually correct for grids.
        valid: optional (n_nodes,) boolean mask. False marks nodes that carry
            no latent data (e.g. land points for an ocean model) and excludes
            them from region selection and from the analysis steps.
    """

    lat: np.ndarray
    lon: np.ndarray
    grid_shape: tuple[int, int] | None = None
    valid: np.ndarray | None = None

    def __post_init__(self):
        self.lat = np.asarray(self.lat, dtype=float).reshape(-1)
        self.lon = np.asarray(self.lon, dtype=float).reshape(-1)
        if self.lat.shape != self.lon.shape:
            raise ValueError(
                f"lat and lon must have the same length, got {self.lat.shape} "
                f"and {self.lon.shape}"
            )
        # Normalise longitudes to -180 -> 180, which is what the plotting and
        # the haversine selection in utils.py expect.
        self.lon = ((self.lon + 180.0) % 360.0) - 180.0
        if self.valid is not None:
            self.valid = np.asarray(self.valid, dtype=bool).reshape(-1)
            if self.valid.shape != self.lat.shape:
                raise ValueError(
                    "valid mask must have one entry per node, got "
                    f"{self.valid.shape} for {self.lat.shape} nodes"
                )
        if self.grid_shape is not None:
            n_lat, n_lon = self.grid_shape
            if n_lat * n_lon != self.lat.size:
                raise ValueError(
                    f"grid_shape {self.grid_shape} does not match {self.lat.size} nodes"
                )

    @property
    def n_nodes(self) -> int:
        return int(self.lat.size)


# ----------------------------------------------------
# --- 2. The structure registry
# ----------------------------------------------------


@dataclasses.dataclass
class LatentStructure:
    """One latent data layout, plus the callables that read it.

    Attributes:
        name: key used in ``MODELS[...]["latent"]["structure"]``.
        available_months: (model_cfg) -> sorted list of (year, month).
        available_times: (year, month, model_cfg) -> pandas DatetimeIndex.
        load_latent: (flt_time, model_cfg, batch_num, use_translator) ->
            (latent (n_steps, n_nodes, latent_dim), step_labels, time_index).
        node_geometry: (model_cfg) -> NodeGeometry.
        step_name: what one entry of the step axis is called in the UI, e.g.
            "Processor step" for GraphCast, "U-Net level" for Samudra.
        supports_translator: whether the translator checkbox is meaningful.
        ragged_channels: True when channel k is not the same feature at every
            step (U-Nets). The app then warns that cross-step comparison of a
            channel index is not meaningful.
        description: one-line summary shown in the sidebar.
    """

    name: str
    available_months: Callable
    available_times: Callable
    load_latent: Callable
    node_geometry: Callable
    step_name: str = "Processor step"
    supports_translator: bool = False
    ragged_channels: bool = False
    description: str = ""


STRUCTURES: dict[str, LatentStructure] = {}


def register_structure(structure: LatentStructure) -> LatentStructure:
    """Add a structure to the registry, keyed by ``structure.name``."""
    if structure.name in STRUCTURES:
        raise ValueError(f"Latent structure {structure.name!r} is already registered")
    STRUCTURES[structure.name] = structure
    return structure


def get_structure(name: str) -> LatentStructure:
    """Look up a registered structure, with a helpful error if it is missing."""
    try:
        return STRUCTURES[name]
    except KeyError:
        known = ", ".join(sorted(STRUCTURES)) or "(none registered)"
        raise ValueError(
            f"Unknown latent structure: {name!r}. Registered structures: {known}. "
            "Add a new one with latent_structures.register_structure()."
        ) from None


def structure_for(model_cfg: dict) -> LatentStructure:
    """Convenience wrapper: the structure used by a model config entry."""
    return get_structure(model_cfg["latent"]["structure"])


# ----------------------------------------------------
# --- 3. Build a NodeGeometry from grid coordinates
# ----------------------------------------------------


def grid_node_geometry(lat, lon, valid=None) -> NodeGeometry:
    """Flatten a regular or curvilinear lat/lon grid into nodes.

    Args:
        lat: 1-D latitudes (n_lat,) or 2-D (n_lat, n_lon).
        lon: 1-D longitudes (n_lon,) or 2-D (n_lat, n_lon).
        valid: optional 2-D (n_lat, n_lon) boolean mask, e.g. an ocean wet mask.

    Returns:
        NodeGeometry with nodes in row-major (lat outer, lon inner) order, which
        is the order produced by ``np.ravel`` on a (lat, lon) field.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat.ndim == 1 and lon.ndim == 1:
        grid_shape = (lat.size, lon.size)
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise ValueError(
                f"2-D lat/lon must have the same shape, got {lat.shape} and {lon.shape}"
            )
        grid_shape = lat.shape
        lat2d, lon2d = lat, lon
    else:
        raise ValueError(
            "lat and lon must both be 1-D or both 2-D, got ndim "
            f"{lat.ndim} and {lon.ndim}"
        )

    valid_flat = None
    if valid is not None:
        valid_flat = np.asarray(valid, dtype=bool).reshape(-1)

    return NodeGeometry(
        lat=lat2d.reshape(-1),
        lon=lon2d.reshape(-1),
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        valid=valid_flat,
    )


# ----------------------------------------------------
# --- 4. Read coordinates from a file
# ----------------------------------------------------

_LAT_NAMES = ("lat", "latitude", "grid_yt", "yh", "y")
_LON_NAMES = ("lon", "longitude", "grid_xt", "xh", "x")
_MASK_NAMES = ("wet", "mask", "ocean_mask", "wet_mask", "valid")


def _first_present(container, names):
    for name in names:
        if name in container:
            return name
    return None


def load_coords_file(path: str, mask_key: str | None = None):
    """Read (lat, lon, valid) from a coordinates file.

    Supports:
        * ``.npz`` with arrays named lat/latitude/... and lon/longitude/...,
          optionally a mask named wet/mask/ocean_mask/valid.
        * ``.npy`` holding a (2, n) or (n, 2) array of [lat, lon].
        * ``.nc`` / ``.zarr`` read through xarray, using its coordinates.

    Returns:
        (lat, lon, valid) where valid may be None. lat/lon keep their file
        dimensionality (1-D or 2-D) so the caller can decide how to flatten.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Coordinates file not found: {path}")

    lower = path.lower()

    if lower.endswith(".npz"):
        with np.load(path, allow_pickle=False) as npz:
            keys = list(npz.files)
            lat_key = _first_present(keys, _LAT_NAMES)
            lon_key = _first_present(keys, _LON_NAMES)
            if lat_key is None or lon_key is None:
                raise KeyError(
                    f"{path} must contain latitude and longitude arrays; "
                    f"found {keys}"
                )
            lat = np.asarray(npz[lat_key])
            lon = np.asarray(npz[lon_key])
            valid = None
            key = mask_key or _first_present(keys, _MASK_NAMES)
            if key is not None and key in keys:
                valid = np.asarray(npz[key]).astype(bool)
        return lat, lon, valid

    if lower.endswith(".npy"):
        arr = np.asarray(np.load(path, allow_pickle=False))
        if arr.ndim != 2:
            raise ValueError(
                f"{path} must hold a 2-D array of latitudes and longitudes, "
                f"got shape {arr.shape}"
            )
        if arr.shape[0] == 2:
            return arr[0], arr[1], None
        if arr.shape[1] == 2:
            return arr[:, 0], arr[:, 1], None
        raise ValueError(
            f"{path} must have a dimension of length 2 (lat, lon), got {arr.shape}"
        )

    if lower.endswith(".nc") or lower.endswith(".zarr"):
        import xarray as xr  # imported lazily: only needed for netCDF/zarr coords

        engine = "zarr" if lower.endswith(".zarr") else None
        ds = xr.open_dataset(path, engine=engine)
        try:
            lat_key = _first_present(ds.variables, _LAT_NAMES)
            lon_key = _first_present(ds.variables, _LON_NAMES)
            if lat_key is None or lon_key is None:
                raise KeyError(
                    f"{path} must contain latitude and longitude variables; "
                    f"found {list(ds.variables)}"
                )
            lat = np.asarray(ds[lat_key].values)
            lon = np.asarray(ds[lon_key].values)
            valid = None
            key = mask_key or _first_present(ds.variables, _MASK_NAMES)
            if key is not None and key in ds.variables:
                valid = np.asarray(ds[key].values).astype(bool)
        finally:
            ds.close()
        return lat, lon, valid

    raise ValueError(
        f"Unsupported coordinates file type: {path}. Use .npz, .npy, .nc or .zarr."
    )


# ----------------------------------------------------
# --- 5. Reorder an on-disk latent array
# ----------------------------------------------------

_CANONICAL_DIMS = ("time", "batch", "step", "channel", "lat", "lon")
_DIM_ALIASES = {
    "t": "time",
    "time": "time",
    "b": "batch",
    "batch": "batch",
    "step": "step",
    "block": "step",
    "level": "step",
    "layer": "step",
    "c": "channel",
    "ch": "channel",
    "channel": "channel",
    "chan": "channel",
    "feature": "channel",
    "latent": "channel",
    "y": "lat",
    "lat": "lat",
    "latitude": "lat",
    "x": "lon",
    "lon": "lon",
    "longitude": "lon",
    "node": "node",
    "nodes": "node",
    "mesh": "node",
}


def parse_dims(dims) -> list[str]:
    """Normalise a dims specification into canonical dimension names.

    ``dims`` may be a comma-separated string ("time,batch,channel,lat,lon"),
    a compact string ("tbchw" is not supported; use commas), or a sequence.
    """
    if isinstance(dims, str):
        parts = [p.strip() for p in dims.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in dims]
    out = []
    for part in parts:
        key = part.lower()
        if key not in _DIM_ALIASES:
            raise ValueError(
                f"Unknown latent dimension {part!r}. Use one of: "
                f"time, batch, channel, lat, lon, node."
            )
        out.append(_DIM_ALIASES[key])
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate dimension in dims specification: {dims!r}")
    return out


def transpose_to_canonical(arr: np.ndarray, dims) -> tuple[np.ndarray, list[str]]:
    """Reorder ``arr`` so its dimensions appear in canonical order.

    Canonical order is (time, batch, step, channel, lat, lon), or (time, batch,
    step, node, channel) for unstructured data; dimensions absent from ``dims``
    are simply absent from the result.

    Returns:
        (transposed array, canonical dimension names in order).
    """
    names = parse_dims(dims)
    if len(names) != arr.ndim:
        raise ValueError(
            f"dims {names} describes {len(names)} dimensions but the array has "
            f"{arr.ndim} (shape {arr.shape})"
        )
    if "node" in names:
        order = [d for d in ("time", "batch", "step", "node", "channel") if d in names]
    else:
        order = [d for d in _CANONICAL_DIMS if d in names]
    if len(order) != len(names):
        missing = set(names) - set(order)
        raise ValueError(f"Cannot order dimensions {missing}")
    axes = [names.index(d) for d in order]
    return np.transpose(arr, axes), order


def select_time_batch(
    arr: np.ndarray, dims, time_index: int, batch_num: int
) -> tuple[np.ndarray, list[str]]:
    """Take one timestep and one batch member, returning the remaining dims."""
    arr, order = transpose_to_canonical(arr, dims)
    remaining = list(order)
    if "time" in remaining:
        n_time = arr.shape[remaining.index("time")]
        if not 0 <= time_index < n_time:
            raise IndexError(
                f"Time index {time_index} is out of range for a latent file with "
                f"{n_time} timesteps"
            )
        arr = arr[time_index]
        remaining.remove("time")
    if "batch" in remaining:
        n_batch = arr.shape[remaining.index("batch")]
        if not 0 <= batch_num < n_batch:
            raise IndexError(
                f"Batch index {batch_num} is out of range for a latent file with "
                f"{n_batch} batch members"
            )
        arr = arr[(slice(None),) * remaining.index("batch") + (batch_num,)]
        remaining.remove("batch")
    return arr, remaining


# ----------------------------------------------------
# --- 6. Flatten a gridded block into nodes
# ----------------------------------------------------


def grid_latent_to_nodes(arr: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Convert one step's latent block into (n_nodes, latent_dim).

    Args:
        arr: array whose dimensions are described by ``order``, which must be
            either ("channel", "lat", "lon") or ("node", "channel").
        order: canonical dimension names of ``arr``.
    """
    order = list(order)
    if order == ["node", "channel"]:
        return np.asarray(arr)
    if order == ["channel", "lat", "lon"]:
        n_channel, n_lat, n_lon = arr.shape
        # (channel, lat, lon) -> (lat*lon, channel), lat outer / lon inner so it
        # matches grid_node_geometry's row-major flattening.
        return np.asarray(arr).reshape(n_channel, n_lat * n_lon).T
    raise ValueError(
        "Expected a latent block with dimensions (channel, lat, lon) or "
        f"(node, channel) after selecting time and batch, got {order}"
    )


# ----------------------------------------------------
# --- 7. Nearest-neighbour resample of coarse levels
# ----------------------------------------------------


def resample_grid_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample a (channel, lat, lon) block onto a new grid.

    U-Net style models (Samudra) hold their deeper levels on coarser grids. To
    show every level on the same map and to compare levels within one region,
    the coarse levels are expanded back onto the model's native output grid.
    Nearest neighbour is used deliberately: it does not invent intermediate
    activation values the model never produced.
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a (channel, lat, lon) block, got shape {arr.shape}")
    _, n_lat, n_lon = arr.shape
    target_lat, target_lon = target_shape
    if (n_lat, n_lon) == (target_lat, target_lon):
        return arr
    lat_idx = np.minimum((np.arange(target_lat) * n_lat) // target_lat, n_lat - 1)
    lon_idx = np.minimum((np.arange(target_lon) * n_lon) // target_lon, n_lon - 1)
    return arr[:, lat_idx][:, :, lon_idx]


# ----------------------------------------------------
# --- 8. Stack steps with different channel widths
# ----------------------------------------------------


def stack_ragged_steps(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-step (n_nodes, latent_dim_step) blocks into one array.

    Steps with fewer channels than the widest step are padded with NaN. The app
    treats a channel as absent at a step when it is not finite there, so NaN
    padding keeps ragged architectures usable without pretending the missing
    channels are zero.

    Returns:
        (n_steps, n_nodes, max_latent_dim) float array.
    """
    if not blocks:
        raise ValueError("No latent blocks to stack")
    n_nodes = blocks[0].shape[0]
    for i, block in enumerate(blocks):
        if block.shape[0] != n_nodes:
            raise ValueError(
                f"Step {i} has {block.shape[0]} nodes but step 0 has {n_nodes}; "
                "all steps must be resampled onto the same nodes first"
            )
    max_dim = max(block.shape[1] for block in blocks)
    out = np.full((len(blocks), n_nodes, max_dim), np.nan, dtype=np.float32)
    for i, block in enumerate(blocks):
        out[i, :, : block.shape[1]] = block
    return out


# ----------------------------------------------------
# --- 9. Filename scanning shared by the structures
# ----------------------------------------------------


def month_file_pattern(template: str) -> re.Pattern:
    """Build a regex from a filename template such as
    ``latent_mesh_step_{step}_{year}_{month}.npz``.

    ``{step}`` matches an integer (the processor step / block / level index),
    ``{year}`` four digits and ``{month}`` two digits.
    """
    escaped = re.escape(template)
    escaped = escaped.replace(re.escape("{step}"), r"(?P<step>\d+)")
    escaped = escaped.replace(re.escape("{year}"), r"(?P<year>\d{4})")
    escaped = escaped.replace(re.escape("{month}"), r"(?P<month>\d{2})")
    return re.compile(rf"^{escaped}$")


def parse_month_files(latent_dir: str, template: str) -> list[dict]:
    """Scan ``latent_dir`` for files matching ``template``.

    Returns:
        A list of {"filename", "path", "step", "year", "month"} dicts sorted by
        (year, month, step).
    """
    if not os.path.isdir(latent_dir):
        raise FileNotFoundError(
            f"Latent directory not found: {latent_dir}. Check paths.json."
        )
    pattern = month_file_pattern(template)
    found = []
    for name in os.listdir(latent_dir):
        match = pattern.match(name)
        if not match:
            continue
        groups = match.groupdict()
        found.append(
            {
                "filename": name,
                "path": os.path.join(latent_dir, name),
                "step": int(groups.get("step", 0)),
                "year": int(groups["year"]),
                "month": int(groups["month"]),
            }
        )
    return sorted(found, key=lambda f: (f["year"], f["month"], f["step"]))


def available_months_from_files(latent_dir: str, template: str) -> list[tuple[int, int]]:
    """Sorted unique (year, month) pairs present in ``latent_dir``."""
    return sorted({(f["year"], f["month"]) for f in parse_month_files(latent_dir, template)})


# ----------------------------------------------------
# --- 10. Time axis helpers shared by the structures
# ----------------------------------------------------


def _month_bounds(year: int, month: int):
    import pandas as pd

    if year is None or month is None:
        raise ValueError(f"Invalid year/month: year={year}, month={month}")
    start_of_month = pd.Timestamp(year, month, 1, tz="UTC")
    if month == 12:
        end_of_month = pd.Timestamp(year + 1, 1, 1, tz="UTC")
    else:
        end_of_month = pd.Timestamp(year, month + 1, 1, tz="UTC")
    return start_of_month, end_of_month


def times_for_month(year: int, month: int, model_cfg: dict, n_times: int | None = None):
    """Build the time axis of one month's latent file.

    The spacing comes from the model config: ``timestep_freq`` (any pandas
    offset alias, e.g. "MS" for monthly data) takes precedence over
    ``timestep_hours``; ``timestep_start_hour`` sets the first timestamp.

    Args:
        n_times: when known (e.g. read from the latent file), the axis is
            truncated to exactly this many timestamps.

    Returns:
        pandas.DatetimeIndex, timezone aware (UTC).
    """
    import pandas as pd

    start_of_month, end_of_month = _month_bounds(year, month)
    start_hour = int(model_cfg.get("timestep_start_hour", 0) or 0)
    start = start_of_month + pd.Timedelta(hours=start_hour)

    freq = model_cfg.get("timestep_freq")
    if freq is None:
        freq = f"{int(model_cfg.get('timestep_hours', 6))}h"

    if n_times is not None:
        return pd.date_range(start=start, periods=int(n_times), freq=freq)
    return pd.date_range(start=start, end=end_of_month, freq=freq, inclusive="left")


def time_index_for(flt_time, model_cfg: dict, times=None) -> int:
    """Index of ``flt_time`` within its month's latent time axis."""
    import pandas as pd

    flt_time = pd.Timestamp(flt_time)
    if times is None:
        times = times_for_month(flt_time.year, flt_time.month, model_cfg)
    times = pd.DatetimeIndex(times)
    if flt_time.tz is None and times.tz is not None:
        flt_time = flt_time.tz_localize(times.tz)
    elif flt_time.tz is not None and times.tz is None:
        flt_time = flt_time.tz_localize(None)
    matches = np.flatnonzero(times == flt_time)
    if matches.size == 0:
        raise ValueError(
            f"{flt_time} is not one of the {len(times)} timestamps available for "
            f"{flt_time.year}-{flt_time.month:02d}"
        )
    return int(matches[0])


def decode_time_array(values):
    """Turn a stored time array (ISO strings, datetime64 or epoch) into an index."""
    import pandas as pd

    arr = np.asarray(values)
    if arr.dtype.kind in ("U", "S", "O"):
        arr = arr.astype(str)
    index = pd.DatetimeIndex(pd.to_datetime(arr, utc=True))
    return index


# ----------------------------------------------------
# --- 11. Reading latent files (.npz / .nc)
# ----------------------------------------------------

_COORD_NAMES = {"time", "lat", "latitude", "lon", "longitude", "step", "block", "level"}

DEFAULT_GRID_DIMS = "time,batch,channel,lat,lon"


def pick_array_key(keys, array_key=None):
    """Choose which array inside an .npz file holds the latents."""
    if array_key is not None:
        if array_key not in keys:
            raise KeyError(f"Array {array_key!r} not found among {list(keys)}")
        return array_key
    candidates = [k for k in keys if k.lower() not in _COORD_NAMES]
    if not candidates:
        raise KeyError(f"No latent array found among {list(keys)}")
    return candidates[0]


def read_latent_file(path: str, latent_cfg: dict):
    """Read one latent file written as .npz or .nc.

    Args:
        path: file to read.
        latent_cfg: the ``latent`` block of a model config. ``array_key``
            selects a named array, ``dims`` describes the dimension order
            (optional for .nc, where the variable's own dimension names are
            used).

    Returns:
        (array, dims, times) where ``times`` is a DatetimeIndex when the file
        stores its own time axis, otherwise None.
    """
    array_key = latent_cfg.get("array_key")
    dims = latent_cfg.get("dims", DEFAULT_GRID_DIMS)

    if path.lower().endswith(".nc"):
        import xarray as xr

        with xr.open_dataset(path) as ds:
            name = array_key or list(ds.data_vars)[0]
            da = ds[name]
            file_dims = latent_cfg.get("dims") or ",".join(da.dims)
            arr = np.asarray(da.values)
            times = decode_time_array(ds["time"].values) if "time" in ds.coords else None
        return arr, file_dims, times

    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        arr = np.asarray(npz[pick_array_key(keys, array_key)])
        times = decode_time_array(npz["time"]) if "time" in keys else None
    return arr, dims, times


def read_grid_from_file(path: str):
    """Read lat/lon coordinates stored alongside latents, or (None, None)."""
    if path.lower().endswith(".nc"):
        import xarray as xr

        with xr.open_dataset(path) as ds:
            lat = next((np.asarray(ds[n].values) for n in _LAT_NAMES if n in ds), None)
            lon = next((np.asarray(ds[n].values) for n in _LON_NAMES if n in ds), None)
        return lat, lon

    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        lat = next((np.asarray(npz[n]) for n in _LAT_NAMES if n in keys), None)
        lon = next((np.asarray(npz[n]) for n in _LON_NAMES if n in keys), None)
    return lat, lon


def regular_global_grid(n_lat: int, n_lon: int):
    """Cell-centre latitudes and longitudes of a regular global grid.

    Matches the equiangular grid ACE uses for its 1-degree configurations:
    latitudes run south to north, longitudes from -180 to 180.
    """
    dlat = 180.0 / n_lat
    dlon = 360.0 / n_lon
    lat = -90.0 + dlat / 2.0 + dlat * np.arange(n_lat)
    lon = -180.0 + dlon / 2.0 + dlon * np.arange(n_lon)
    return lat, lon
