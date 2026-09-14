#!/usr/bin/env python3
"""Build a visualiser reference dataset from an `fme` inference run's output.

Step 2 of the app draws a physical field at the forecast time and the step
before it. The natural source for a model run is the run itself: fme writes
``autoregressive_predictions.nc`` (the predicted state at every step) and
``initial_condition.nc`` (the state the rollout started from). This script
stitches the two into one time series in the layout the app reads:

* dimensions ``(time, lat, lon)``, one ensemble member (``--sample``);
* ``time`` taken from the predictions' ``valid_time``, with the initial
  condition prepended so the first predicted step has a state to difference
  against;
* model years shifted by ``--year-offset``, the same offset
  ``extract_latents_fme.py`` used, so reference times line up with the latents.

Enable ``save_prediction_files: true`` in the inference config's
``data_writer`` to get the predictions, and set a ``seed`` if the model is
stochastic, so the run that produced the latents and the run that produced the
predictions follow the same rollout (or capture both from one run).

Usage::

    python scripts/fme_predictions_to_reference.py run/ocean reference/ocean.nc \\
        --year-offset 1600
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr


def to_pandas_time(value, year_offset):
    """A cftime (or datetime) as a pandas Timestamp, years shifted."""
    return pd.Timestamp(
        year=value.year + year_offset, month=value.month, day=value.day,
        hour=value.hour, minute=value.minute, second=value.second,
    )


def build_reference(run_component_dir, sample=0, year_offset=0, variables=None,
                    max_times=None):
    """Initial condition + predictions for one component, as a (time, lat, lon) dataset."""
    predictions = xr.open_dataset(
        os.path.join(run_component_dir, "autoregressive_predictions.nc"),
        decode_timedelta=False,
    )
    initial = xr.open_dataset(os.path.join(run_component_dir, "initial_condition.nc"))

    names = variables or [
        name for name, da in predictions.data_vars.items()
        if set(da.dims) >= {"lat", "lon"}
    ]
    missing = [name for name in names if name not in initial]
    if missing:
        print(f"note: {len(missing)} variables are not in the initial condition and "
              f"are dropped: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        names = [name for name in names if name in initial]
    if not names:
        raise ValueError("No variables are present in both the predictions and the "
                         "initial condition.")

    valid_time = predictions["valid_time"].isel(sample=sample).values
    n_predicted = len(valid_time) if max_times is None else min(max_times, len(valid_time))
    valid_time = valid_time[:n_predicted]
    init_time = np.asarray(predictions["init_time"].values).ravel()[sample]
    times = [to_pandas_time(t, year_offset) for t in [init_time, *valid_time]]

    arrays = {}
    for name in names:
        first = initial[name].isel(sample=sample).values[None]
        rest = predictions[name].isel(sample=sample, time=slice(0, n_predicted)).values
        arrays[name] = (("time", "lat", "lon"),
                        np.concatenate([first, rest], axis=0).astype(np.float32))

    reference = xr.Dataset(
        arrays,
        coords={
            "time": pd.DatetimeIndex(times),
            "lat": predictions["lat"].values,
            "lon": predictions["lon"].values,
        },
    )
    reference.attrs["source"] = os.path.abspath(run_component_dir)
    reference.attrs["year_offset"] = int(year_offset)
    reference.attrs["model_time_first"] = str(init_time)
    return reference


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_component_dir",
                        help="an fme run's component output directory, e.g. run/ocean "
                             "(for an uncoupled run, the run directory itself)")
    parser.add_argument("output", help="netCDF file to write")
    parser.add_argument("--sample", type=int, default=0, help="ensemble member to keep")
    parser.add_argument("--year-offset", type=int, default=0,
                        help="years added to model dates; match extract_latents_fme.py")
    parser.add_argument("--max-times", type=int, default=None,
                        help="keep only the first N predicted steps (after the "
                             "initial condition)")
    parser.add_argument("--variables", nargs="+", default=None,
                        help="variables to keep (default: every lat-lon field)")
    args = parser.parse_args()

    reference = build_reference(
        args.run_component_dir, args.sample, args.year_offset, args.variables,
        args.max_times,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    reference.to_netcdf(args.output)
    print(f"wrote {args.output}: {len(reference.data_vars)} variables, "
          f"times {[str(t)[:16] for t in reference.time.values]}")


if __name__ == "__main__":
    main()
