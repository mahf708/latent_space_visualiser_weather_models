#!/usr/bin/env python3
"""Generate small synthetic demo data for every supported model structure.

This exists so the visualiser can be run end to end, and so a new structure can
be checked, without downloading any real latent data. The fields are
*synthetic*: they are built from a handful of smooth patterns on the sphere so
that the analyses in the app produce recognisable structure (a region picks out
a few channels, cosine similarity highlights similar regions, PCA finds a small
number of meaningful components). They are not model output and say nothing
about any real model.

Usage::

    python scripts/make_demo_data.py                 # writes ./demo_data
    python scripts/make_demo_data.py --out /tmp/demo --write-paths

Then point ``paths.json`` at the result (``--write-paths`` does it for you) and
run ``streamlit run app.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

RNG_SEED = 0

# Centres of the synthetic "weather features" the latent channels respond to.
FEATURE_CENTRES = [
    (55.0, -15.0),    # North Atlantic
    (10.0, 120.0),    # Maritime continent
    (-35.0, 25.0),    # Southern Africa
    (30.0, -100.0),   # North America
    (-10.0, -60.0),   # Amazon
    (65.0, 90.0),     # Siberia
]


# ----------------------------------------------------
# --- Shared field construction
# ----------------------------------------------------


def feature_fields(lat, lon):
    """Smooth basis patterns evaluated at the given coordinates.

    Returns an (n_features, n_points) array: a Gaussian blob per feature centre
    plus two zonal waves and a latitudinal gradient, so the synthetic latents
    have both local and global structure.
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    fields = []

    for centre_lat, centre_lon in FEATURE_CENTRES:
        dlon = np.deg2rad(((lon - centre_lon + 180.0) % 360.0) - 180.0)
        dlat = np.deg2rad(lat - centre_lat)
        # great-circle-ish distance, good enough for a demo blob
        distance = np.sqrt(dlat**2 + (dlon * np.cos(np.deg2rad(centre_lat))) ** 2)
        fields.append(np.exp(-((distance / 0.35) ** 2)))

    fields.append(np.sin(np.deg2rad(2 * lon)) * np.cos(np.deg2rad(lat)))
    fields.append(np.cos(np.deg2rad(3 * lon)) * np.cos(np.deg2rad(lat)) ** 2)
    fields.append(np.sin(np.deg2rad(lat)))

    return np.stack(fields, axis=0)


def synth_latents(lat, lon, n_steps, n_channels, n_times, rng, step_gain=None):
    """Build (n_times, n_steps, n_channels, n_points) synthetic activations.

    Each channel is a fixed random mixture of the basis patterns, so channels
    are correlated in a low-rank way (which is what makes the PCA step show
    something) and each step re-weights the mixture, so activations evolve
    through the network.
    """
    basis = feature_fields(lat, lon)               # (n_features, n_points)
    n_features, n_points = basis.shape

    mixing = rng.normal(size=(n_channels, n_features))
    # make a handful of channels strongly tied to single features, so the
    # "top activated channels" in a region are interpretable
    for channel in range(min(n_channels, n_features)):
        mixing[channel] *= 0.2
        mixing[channel, channel] = 2.5

    out = np.empty((n_times, n_steps, n_channels, n_points), dtype=np.float32)
    for t in range(n_times):
        # a slow drift of the feature amplitudes through time
        time_weights = 1.0 + 0.3 * np.sin(np.linspace(0, np.pi, n_features) + 0.7 * t)
        for step in range(n_steps):
            gain = step_gain[step] if step_gain is not None else (0.4 + step / n_steps)
            step_weights = time_weights * (1.0 + 0.25 * np.cos(0.6 * step + np.arange(n_features)))
            activations = (mixing * step_weights) @ basis
            noise = rng.normal(scale=0.05, size=(n_channels, n_points))
            out[t, step] = (gain * (activations + noise)).astype(np.float32)
    return out


def reference_dataset(lat, lon, times, rng, ocean_mask=None):
    """A small reference dataset with the variables the app expects."""
    n_lat, n_lon = len(lat), len(lon)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    basis = feature_fields(lat2d, lon2d)

    def field(weights, offset, scale):
        flat = np.tensordot(np.asarray(weights), basis, axes=(0, 0))
        return offset + scale * flat.reshape(n_lat, n_lon)

    n_features = basis.shape[0]
    temperature = np.empty((len(times), n_lat, n_lon), dtype=np.float32)
    humidity = np.empty_like(temperature)

    for i, _ in enumerate(times):
        drift = 1.0 + 0.15 * i
        temperature[i] = field(
            drift * rng.normal(scale=1.0, size=n_features), 288.0, 6.0
        ) - 25.0 * np.abs(np.sin(np.deg2rad(lat)))[:, None]
        humidity[i] = np.clip(
            field(drift * rng.normal(scale=1.0, size=n_features), 0.008, 0.003), 0.0, None
        )

    data_vars = {
        "surface_temperature": (("time", "lat", "lon"), temperature),
        "specific_total_water": (("time", "lat", "lon"), humidity),
    }
    if ocean_mask is not None:
        for name in ("surface_temperature", "specific_total_water"):
            dims, values = data_vars[name]
            values = values.copy()
            values[:, ~ocean_mask] = np.nan
            data_vars[name] = (dims, values)

    return xr.Dataset(
        data_vars,
        coords={"time": pd.DatetimeIndex(times).tz_localize(None), "lat": lat, "lon": lon},
    )


def regular_grid(n_lat, n_lon):
    """Cell centres of a regular global grid, south to north, -180 to 180."""
    dlat, dlon = 180.0 / n_lat, 360.0 / n_lon
    lat = -90.0 + dlat / 2.0 + dlat * np.arange(n_lat)
    lon = -180.0 + dlon / 2.0 + dlon * np.arange(n_lon)
    return lat, lon


# ----------------------------------------------------
# --- GraphCast: unstructured mesh
# ----------------------------------------------------


def fibonacci_sphere(n_points):
    """Roughly even points on a sphere, standing in for an icosahedral mesh."""
    index = np.arange(n_points) + 0.5
    cos_theta = 1.0 - 2.0 * index / n_points
    theta = np.arccos(cos_theta)
    phi = np.pi * (1.0 + 5.0**0.5) * index
    phi = ((phi + np.pi) % (2 * np.pi)) - np.pi
    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi)
    return lat, lon


def write_graphcast(out_dir, rng, n_nodes=2562, n_steps=9, n_channels=32, n_times=4):
    """GraphCast small: one file per processor step, latents on mesh nodes."""
    latent_dir = os.path.join(out_dir, "graphcast_small", "latent_data")
    translator_dir = os.path.join(out_dir, "graphcast_small", "translators")
    reference_dir = os.path.join(out_dir, "graphcast_small", "era5")
    for directory in (latent_dir, translator_dir, reference_dir):
        os.makedirs(directory, exist_ok=True)

    lat, lon = fibonacci_sphere(n_nodes)

    # Mesh node file: [cos(theta), cos(phi), sin(phi)] per node, as the app's
    # mesh_features_to_latlon expects.
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)
    mesh_nodes = np.stack([np.cos(theta), np.cos(phi), np.sin(phi)], axis=1)
    np.save(os.path.join(out_dir, "graphcast_small", "mesh_nodes.npy"), mesh_nodes)

    year, month = 2020, 1
    latents = synth_latents(lat, lon, n_steps, n_channels, n_times, rng)

    for step in range(n_steps):
        # (timesteps, nodes, batch, latent_dim)
        array = np.transpose(latents[:, step], (0, 2, 1))[:, :, None, :]
        np.savez_compressed(
            os.path.join(latent_dir, f"latent_mesh_step_{step}_{year}_{month:02d}.npz"),
            latent=array.astype(np.float32),
        )

    # Translators: near-identity maps, as a translator trained to align an
    # early processor step with the final one would be.
    for step in range(n_steps - 1):
        W = np.eye(n_channels) + 0.05 * rng.normal(size=(n_channels, n_channels))
        b = 0.01 * rng.normal(size=n_channels)
        np.savez_compressed(
            os.path.join(translator_dir, f"translator_matrix_{step}_gnn.npz"),
            W=W.astype(np.float32),
            b=b.astype(np.float32),
        )

    ref_lat, ref_lon = regular_grid(45, 90)
    times = pd.date_range("2020-01-01 12:00", periods=n_times, freq="6h")
    dataset = reference_dataset(ref_lat, ref_lon, times, rng)
    dataset.to_netcdf(
        os.path.join(reference_dir, f"Graphcast_small_processed_input_{year}{month:02d}.nc")
    )

    return {
        "latent_dir": latent_dir,
        "translator_dir": translator_dir,
        "era5_basepath": reference_dir,
        "graph_coords_filepath": os.path.join(out_dir, "graphcast_small", "mesh_nodes.npy"),
    }


# ----------------------------------------------------
# --- ACE: gridded SFNO blocks
# ----------------------------------------------------


def write_ace(out_dir, rng, n_lat=45, n_lon=90, n_blocks=8, n_channels=24, n_times=4):
    """ACE2-style latents: one file per SFNO block, on a regular grid."""
    latent_dir = os.path.join(out_dir, "ace2_era5", "latent_data")
    reference_dir = os.path.join(out_dir, "ace2_era5", "reference")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)

    lat, lon = regular_grid(n_lat, n_lon)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    year, month = 2020, 1
    times = pd.date_range("2020-01-01 00:00", periods=n_times, freq="6h")
    latents = synth_latents(lat2d, lon2d, n_blocks, n_channels, n_times, rng)

    for block in range(n_blocks):
        # (time, batch, channel, lat, lon)
        array = latents[:, block].reshape(n_times, 1, n_channels, n_lat, n_lon)
        np.savez_compressed(
            os.path.join(latent_dir, f"latent_grid_block_{block}_{year}_{month:02d}.npz"),
            latent=array.astype(np.float32),
            lat=lat.astype(np.float32),
            lon=lon.astype(np.float32),
            time=np.array([t.isoformat() for t in times]),
        )

    dataset = reference_dataset(lat, lon, times, rng)
    dataset.to_netcdf(
        os.path.join(reference_dir, f"ace2_era5_reference_{year}{month:02d}.nc")
    )

    return {
        "latent_dir": latent_dir,
        "reference_basepath": reference_dir,
        "reference_filename": "ace2_era5_reference_{ym_str}.nc",
        "processor_step_max": n_blocks - 1,
    }


# ----------------------------------------------------
# --- Samudra: ocean U-Net levels
# ----------------------------------------------------


def synthetic_ocean_mask(lat2d, lon2d):
    """A crude land/ocean mask, so the demo exercises masking on a real shape."""
    land = np.zeros(lat2d.shape, dtype=bool)
    blobs = [
        (45.0, -100.0, 28.0, 34.0),   # North America
        (-12.0, -58.0, 22.0, 18.0),   # South America
        (10.0, 20.0, 30.0, 25.0),     # Africa
        (55.0, 80.0, 26.0, 70.0),     # Eurasia
        (-25.0, 134.0, 13.0, 20.0),   # Australia
        (-88.0, 0.0, 6.0, 180.0),     # Antarctica
    ]
    for centre_lat, centre_lon, half_lat, half_lon in blobs:
        dlon = np.abs(((lon2d - centre_lon + 180.0) % 360.0) - 180.0)
        land |= (np.abs(lat2d - centre_lat) < half_lat) & (dlon < half_lon)
    return ~land


def write_samudra(out_dir, rng, n_lat=45, n_lon=90, n_months=4):
    """Samudra-style latents: one file per U-Net level, coarsening with depth."""
    latent_dir = os.path.join(out_dir, "samudra_ocean", "latent_data")
    reference_dir = os.path.join(out_dir, "samudra_ocean", "reference")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)

    lat, lon = regular_grid(n_lat, n_lon)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    wet = synthetic_ocean_mask(lat2d, lon2d)

    np.savez_compressed(
        os.path.join(out_dir, "samudra_ocean", "ocean_grid.npz"),
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        wet=wet,
    )

    # Encoder halves the resolution at each level and widens the channels;
    # the decoder mirrors it. Matches Samudra's default ch_width shape.
    level_shapes = [(n_lat, n_lon), (n_lat // 2, n_lon // 2), (n_lat // 4, n_lon // 4),
                    (n_lat // 2, n_lon // 2), (n_lat, n_lon)]
    level_channels = [16, 20, 24, 20, 16]

    times = pd.date_range("2020-01-01", periods=n_months, freq="MS")

    for month_index, timestamp in enumerate(times):
        for level, ((level_lat, level_lon), n_channels) in enumerate(
            zip(level_shapes, level_channels, strict=True)
        ):
            level_lat_vals, level_lon_vals = regular_grid(level_lat, level_lon)
            grid_lat, grid_lon = np.meshgrid(level_lat_vals, level_lon_vals, indexing="ij")
            latents = synth_latents(
                grid_lat, grid_lon, 1, n_channels, 1, np.random.default_rng(RNG_SEED + level)
            )
            # (time, batch, channel, lat, lon), one month per file
            array = latents[:, 0].reshape(1, 1, n_channels, level_lat, level_lon)
            array = array * (1.0 + 0.1 * month_index)
            np.savez_compressed(
                os.path.join(
                    latent_dir,
                    f"latent_level_{level}_{timestamp.year}_{timestamp.month:02d}.npz",
                ),
                latent=array.astype(np.float32),
                time=np.array([timestamp.isoformat()]),
            )

    dataset = reference_dataset(lat, lon, times, rng, ocean_mask=wet)
    dataset.to_netcdf(os.path.join(reference_dir, "samudra_reference.nc"))

    return {
        "latent_dir": latent_dir,
        "reference_basepath": reference_dir,
        "reference_filename": "samudra_reference.nc",
        "reference_label": "Ocean reanalysis",
        "grid_coords_filepath": os.path.join(out_dir, "samudra_ocean", "ocean_grid.npz"),
        "processor_step_max": len(level_shapes) - 1,
    }


# ----------------------------------------------------
# --- Entry point
# ----------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="demo_data", help="output directory")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["graphcast", "ace", "samudra"],
        choices=["graphcast", "ace", "samudra"],
        help="which demo datasets to write",
    )
    parser.add_argument(
        "--write-paths",
        action="store_true",
        help="also write paths.json pointing at the generated data",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    paths = {}
    if "graphcast" in args.models:
        print("writing GraphCast demo data...")
        paths["graphcast_small"] = write_graphcast(out_dir, rng)
    if "ace" in args.models:
        print("writing ACE demo data...")
        paths["ace2_era5"] = write_ace(out_dir, rng)
    if "samudra" in args.models:
        print("writing Samudra demo data...")
        paths["samudra_ocean"] = write_samudra(out_dir, rng)

    print(f"demo data written to {out_dir}")

    if args.write_paths:
        with open("paths.json", "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=2)
            f.write("\n")
        print("wrote paths.json")
    else:
        print("\nAdd this to paths.json:\n")
        print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
