#!/usr/bin/env python3
r"""Extract latent activations from an `fme` (ACE) model run, for the visualiser.

The ``fme`` package (https://github.com/mahf708/ace) does not write internal
activations during inference, so this script attaches forward hooks to a
network's blocks, runs fme's own inference loop, and writes what the hooks
captured in the layout the app's structure modules expect:

* ``--arch sfno``    -> ``latent_grid_block_<block>_<year>_<month>.npz``,
                        read by ``ace_structure_1`` (ACE, ACE2, any SFNO).
* ``--arch samudra`` -> ``latent_level_<level>_<year>_<month>.npz``,
                        read by ``samudra_structure_1``. Only the ConvNeXt
                        blocks are hooked, so you get one activation per U-Net
                        level rather than one per pooling and upsampling step;
                        ``--module-types any`` keeps all of them.

Each file holds ``latent`` with dimensions ``(time, batch, channel, lat, lon)``
and a ``time`` array of ISO timestamps. ``--write-grid`` additionally saves the
component's latitudes, longitudes and land/ocean mask as a ``.npz`` you can
point ``grid_coords_filepath`` at.

Config types
------------
The script reads whichever config you already run the model with, and works out
which of fme's four entry points it belongs to:

================  ==================================  =========================
Config has        Entry point                         Component
================  ==================================  =========================
n_forward_steps   fme.ace.inference.*                 the single network
n_coupled_steps   fme.coupled.inference.*             pick with --component
================  ==================================  =========================

and, within each, ``loader`` means an evaluator config while
``initial_condition`` + ``forcing_loader`` means an inference config.

**Coupled checkpoints (SamudrACE) need --component.** A coupled stepper holds
two networks - an SFNO atmosphere and a Samudra ocean - and there is no sensible
default, so the script refuses to guess.

Requirements
------------
``torch`` and ``fme``, which the Streamlit app itself does not need and which
are therefore not in ``requirements.txt``::

    pip install fme

Usage
-----
Start from a config that already runs for your checkpoint (see
https://ai2-climate-emulator.readthedocs.io/en/latest/quickstart.html), then::

    # check which modules would be hooked before committing to a long run
    python scripts/extract_latents_fme.py config-inference.yaml \
        --component ocean --list-modules

    python scripts/extract_latents_fme.py config-inference.yaml \
        --component ocean \
        --out my_data/samudrace/ocean/latent_data \
        --write-grid my_data/samudrace/ocean_grid.npz \
        --max-times 6

Notes
-----
* Activations are written raw: nothing is normalised or rescaled. Magnitudes are
  comparable across steps only where the architecture shares one latent space
  (SFNO blocks do; U-Net levels do not, which is why the app marks Samudra
  ragged).
* Memory is the limit, during extraction and again in the app, which pads
  every level to the widest and resamples coarse levels onto the output grid.
  On SamudrACE-E3SMv3 at 1 degree, one timestep is 1.2 GB for the ocean (nine
  levels, up to 520 channels) and 0.8 GB for the atmosphere (eight blocks of
  384). Use ``--max-times``, ``--dtype float16``, and where you only want some
  levels, a narrower ``--module-pattern`` such as
  ``'^(module\.)?layers\.(0|8|16)$'``.
* The inference run itself proceeds normally and still writes its usual output
  to the config's ``experiment_dir``.
* Deriving the first timestamp costs one extra checkpoint load. Pass
  ``--start-time`` to skip that; it is required for evaluator configs, which
  have no initial-condition block to read it from.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import numpy as np

ARCHS = {
    # SFNO processor blocks: fme.ace.models.modulus.sfnonet / makani.sfnonet.
    # Every entry of `blocks` is a processor block outputting
    # (batch, embed_dim, lat, lon), so the name alone is enough. The stochastic
    # NoiseConditionedSFNO (e.g. SamudrACE's atmosphere) wraps the network, so
    # its blocks sit one level down, at conditional_model.blocks.N.
    "sfno": {
        "pattern": r"^(module\.)?(conditional_model\.)?blocks\.\d+$",
        "types": None,
        "template": "latent_grid_block_{step}_{year}_{month:02d}.npz",
    },
    # Samudra's `layers` interleaves ConvNeXt blocks with AvgPool, upsample and
    # a final Conv2d that emits the prediction rather than a latent. Filtering
    # to ConvNeXtBlock gives exactly one activation per U-Net level: with the
    # default ch_width [200, 250, 300, 400] that is 9 levels (4 encoder, a
    # bottleneck, 3 decoder, and the output-width block).
    "samudra": {
        "pattern": r"^(module\.)?layers\.\d+$",
        "types": ("ConvNeXtBlock",),
        "template": "latent_level_{step}_{year}_{month:02d}.npz",
    },
}

DEFAULT_PATTERNS = {name: spec["pattern"] for name, spec in ARCHS.items()}
DEFAULT_TEMPLATES = {name: spec["template"] for name, spec in ARCHS.items()}

# Which arch a coupled component is, absent an explicit --arch.
COMPONENT_ARCH = {"atmosphere": "sfno", "ocean": "samudra"}


# ----------------------------------------------------
# --- Config loading
# ----------------------------------------------------


def detect_config_kind(config_data):
    """Work out which of fme's four entry points a config belongs to.

    Returns:
        (coupling, mode) where coupling is "coupled" or "single" and mode is
        "inference" or "evaluator".
    """
    if "n_coupled_steps" in config_data:
        coupling = "coupled"
    elif "n_forward_steps" in config_data:
        coupling = "single"
    else:
        raise ValueError(
            "Config has neither n_coupled_steps nor n_forward_steps, so it is "
            "not an fme inference or evaluator config."
        )

    if "initial_condition" in config_data and "forcing_loader" in config_data:
        mode = "inference"
    elif "loader" in config_data:
        mode = "evaluator"
    else:
        raise ValueError(
            "Config has neither an 'initial_condition' + 'forcing_loader' pair "
            "(inference) nor a 'loader' (evaluator)."
        )
    return coupling, mode


def entry_point(coupling, mode):
    """The config class and run function for one kind of config."""
    if coupling == "coupled":
        if mode == "inference":
            from fme.coupled.inference import inference as module

            return module.InferenceConfig, module.run_inference_from_config, module
        from fme.coupled.inference import evaluator as module

        return module.InferenceEvaluatorConfig, module.run_evaluator_from_config, module

    if mode == "inference":
        from fme.ace.inference import inference as module

        return module.InferenceConfig, module.run_inference_from_config, module
    from fme.ace.inference import evaluator as module

    return module.InferenceEvaluatorConfig, module.run_evaluator_from_config, module


def load_config(yaml_path):
    """Load an fme config, returning it with its kind and raw dict."""
    import dacite
    from fme.core.cli import prepare_config

    config_data = prepare_config(yaml_path)
    coupling, mode = detect_config_kind(config_data)
    config_class, run_function, module = entry_point(coupling, mode)
    config = dacite.from_dict(
        data_class=config_class,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    return config, config_data, coupling, mode, run_function, module


# ----------------------------------------------------
# --- Picking the network and its modules
# ----------------------------------------------------


def component_stepper(stepper, coupling, component):
    """The Stepper for the requested component of a (possibly coupled) stepper."""
    if coupling != "coupled":
        return stepper
    # CoupledStepper.modules is [*atmosphere.modules, *ocean.modules], so
    # indexing it would silently hand back the atmosphere for an ocean request.
    return getattr(stepper, component)


def network_of(stepper):
    """The underlying network of an fme Stepper."""
    modules = list(stepper.modules)
    if not modules:
        raise RuntimeError("The stepper contains no modules.")
    if len(modules) > 1:
        print(f"note: this stepper has {len(modules)} modules; hooking the first")
    return modules[0]


def select_modules(model, pattern, types=None):
    """(name, module) pairs whose qualified name matches ``pattern``.

    Args:
        pattern: regex matched against ``model.named_modules()`` names.
        types: optional class names to keep. A U-Net's layer list mixes the
            blocks you want with pooling, upsampling and output layers, which
            the name alone cannot tell apart.

    Sorted by the trailing integer so step 10 does not sort before step 2.
    """
    regex = re.compile(pattern)
    selected = [(name, mod) for name, mod in model.named_modules() if regex.match(name)]
    if types is not None:
        by_name = selected
        selected = [
            (name, mod) for name, mod in selected if type(mod).__name__ in types
        ]
        if not selected and by_name:
            found = sorted({type(mod).__name__ for _, mod in by_name})
            raise ValueError(
                f"{len(by_name)} modules matched {pattern!r} but none is one of "
                f"{list(types)}; they are {found}. Pass --module-types to choose, "
                "or --module-pattern to match differently."
            )
    if not selected:
        available = [name for name, _ in model.named_modules() if name][:40]
        raise ValueError(
            f"No modules matched {pattern!r}. The first module names are:\n  "
            + "\n  ".join(available)
            + "\nPass --module-pattern with a regex that matches the blocks you want."
        )

    def step_of(name):
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else 0

    return sorted(selected, key=lambda item: step_of(item[0]))


def to_numpy(output, dtype):
    """Detach a hooked output to a numpy array of the requested dtype."""
    import torch

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Hooked output is a {type(output).__name__}, not a tensor. "
            "Hook a different module with --module-pattern."
        )
    return output.detach().to("cpu", dtype=torch.float32).numpy().astype(dtype)


# ----------------------------------------------------
# --- Timestep of the hooked component
# ----------------------------------------------------


def component_timestep(stepper, coupling, component):
    """How much model time passes between two activations of this component.

    ``stepper`` is the full stepper as loaded - the CoupledStepper for a coupled
    config, not the component's own Stepper.

    For a coupled stepper the ocean steps once per coupled step and the
    atmosphere ``n_inner_steps`` times within it, so they have different
    spacings.
    """
    import pandas as pd

    if coupling != "coupled":
        return pd.Timedelta(stepper.training_dataset_info.timestep)
    if component == "ocean":
        return pd.Timedelta(stepper.config.ocean_timestep)
    return pd.Timedelta(stepper.config.atmosphere_timestep)


def component_dataset_info(stepper, coupling, component):
    """The DatasetInfo describing the hooked component's grid."""
    info = stepper.training_dataset_info
    if coupling != "coupled":
        return info
    return getattr(info, component)


# ----------------------------------------------------
# --- Writing output
# ----------------------------------------------------


def write_grid_file(dataset_info, path):
    """Save the component's lat/lon and land/ocean mask for the visualiser.

    The mask comes from fme's spatial mask provider, where 0 means the point is
    masked out. It is written as a boolean ``wet`` array, which is one of the
    names ``latent_structures.load_coords_file`` looks for.
    """
    import numpy as np

    try:
        coordinates = dataset_info.horizontal_coordinates
    except Exception as error:  # fme raises MissingDatasetInfo
        raise RuntimeError(
            f"The checkpoint does not record horizontal coordinates ({error}). "
            "Build the grid file from your dataset instead."
        ) from None

    if not hasattr(coordinates, "lat") or not hasattr(coordinates, "lon"):
        raise RuntimeError(
            f"{type(coordinates).__name__} is not a latitude-longitude grid. The "
            "visualiser has no structure module for it yet."
        )

    lat = coordinates.lat.cpu().numpy()
    lon = coordinates.lon.cpu().numpy()

    wet = None
    provider = getattr(dataset_info, "spatial_mask_provider", None)
    masks = dict(getattr(provider, "masks", {}) or {})
    if masks:
        # mask_2d is the catch-all; mask_0 is the surface level of a 3-D field.
        for key in ("mask_2d", "mask_0"):
            if key in masks:
                wet = masks[key]
                break
        if wet is None:
            key = sorted(masks)[0]
            print(f"note: using {key} as the ocean mask")
            wet = masks[key]
        wet = wet.cpu().numpy() != 0

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arrays = {"lat": lat.astype(np.float32), "lon": lon.astype(np.float32)}
    if wet is not None:
        arrays["wet"] = wet
    np.savez_compressed(path, **arrays)
    shape = wet.shape if wet is not None else (lat.shape, lon.shape)
    print(f"wrote {path}  lat={lat.shape} lon={lon.shape} mask={shape if wet is not None else 'none'}")


def _as_cftime(value, calendar="standard"):
    """A cftime datetime for a cftime, datetime, pandas or ISO-string value."""
    import cftime

    if isinstance(value, cftime.datetime):
        return value
    if isinstance(value, str):
        match = re.match(
            r"^(-?\d{1,5})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?", value
        )
        if not match:
            raise ValueError(f"Cannot parse {value!r} as YYYY-MM-DDTHH:MM:SS")
        parts = [int(p) if p else 0 for p in match.groups()]
        return cftime.datetime(*parts, calendar=calendar)
    return cftime.datetime(
        value.year, value.month, value.day, value.hour, value.minute, value.second,
        calendar=calendar,
    )


def auto_year_offset(year):
    """0 when pandas can represent ``year``, else a round shift into 2000-2099.

    Climate-model control runs use model years like 0425, far outside pandas'
    1677-2262 nanosecond range. Shifting by whole centuries keeps the model
    date readable (0425 becomes 2025) and the app subtracts the offset again
    when it displays times.
    """
    if 1678 <= year <= 2261:
        return 0
    return 2000 - (year // 100) * 100


def model_times(first, step_delta, n_times, year_offset):
    """Timestamps for ``n_times`` steps, stepped in the model's own calendar.

    Stepping happens in cftime (so a no-leap calendar never gains a Feb 29),
    and only then are the dates relabelled into pandas with ``year_offset``
    added. Returns (pandas timestamps, model-calendar ISO strings).
    """
    import pandas as pd

    delta = pd.Timedelta(step_delta).to_pytimedelta()
    shifted, original = [], []
    for i in range(n_times):
        t = first + i * delta
        original.append(t.isoformat())
        shifted.append(
            pd.Timestamp(
                year=t.year + year_offset, month=t.month, day=t.day,
                hour=t.hour, minute=t.minute, second=t.second,
            )
        )
    return shifted, original


def _group_by_month(times):
    """[( (year, month), positions ), ...] preserving time order."""
    groups = defaultdict(list)
    for position, timestamp in enumerate(times):
        groups[(timestamp.year, timestamp.month)].append(position)
    return [(key, np.array(positions)) for key, positions in sorted(groups.items())]


def write_latents(captured, times, out_dir, template, dtype, model_time=None,
                  year_offset=0, calendar=None):
    """Write one file per hooked step, split by month, as the loaders expect.

    ``times`` are what the app reads (pandas-representable, year-shifted if
    needed); ``model_time`` keeps the dates in the model's own calendar.
    """
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    times = pd.DatetimeIndex(times)

    for step, arrays in sorted(captured.items()):
        stacked = np.stack(arrays, axis=0)  # (time, batch, channel, lat, lon)
        if stacked.ndim != 5:
            raise ValueError(
                f"Step {step} activations have shape {stacked.shape}, expected "
                "(time, batch, channel, lat, lon). Hook a different module, or "
                "reshape before writing."
            )
        if len(times) != stacked.shape[0]:
            raise ValueError(
                f"Captured {stacked.shape[0]} timesteps for step {step} but built "
                f"{len(times)} timestamps; pass --start-time to set the first "
                "timestamp explicitly."
            )

        for (year, month), positions in _group_by_month(times):
            path = os.path.join(out_dir, template.format(step=step, year=year, month=month))
            extra = {}
            if model_time is not None:
                extra["model_time"] = np.array(model_time)[positions]
                extra["year_offset"] = np.array(year_offset)
                extra["calendar"] = np.array(calendar or "standard")
            np.savez_compressed(
                path,
                latent=stacked[positions].astype(dtype),
                time=np.array([t.isoformat() for t in times[positions]]),
                **extra,
            )
            print(f"wrote {path}  shape={stacked[positions].shape}")


# ----------------------------------------------------
# --- Entry point
# ----------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="fme inference or evaluator YAML config")
    parser.add_argument("--out", default="latent_data",
                        help="directory to write latent files into")
    parser.add_argument("--component", choices=["atmosphere", "ocean"], default=None,
                        help="which network of a coupled (SamudrACE) checkpoint to "
                             "hook; required for coupled configs")
    parser.add_argument("--arch", choices=sorted(DEFAULT_PATTERNS), default=None,
                        help="preset of module names and output filenames; defaults "
                             "to sfno for the atmosphere and samudra for the ocean")
    parser.add_argument("--module-pattern", default=None,
                        help="regex over model.named_modules() names, overriding --arch")
    parser.add_argument("--module-types", nargs="+", default=None, metavar="CLASS",
                        help="keep only modules of these class names (e.g. "
                             "ConvNeXtBlock); pass 'any' to keep all matches")
    parser.add_argument("--filename-template", default=None,
                        help="output filename template, overriding --arch")
    parser.add_argument("--write-grid", default=None, metavar="PATH",
                        help="also write the component's lat/lon and ocean mask to "
                             "this .npz, for grid_coords_filepath in paths.json")
    parser.add_argument("--max-times", type=int, default=8,
                        help="stop capturing after this many forward passes "
                             "(the inference run itself continues)")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="dtype to store activations in")
    parser.add_argument("--start-time", default=None,
                        help="ISO timestamp of the first captured step; by default "
                             "this is derived from an inference config's initial "
                             "condition, at the cost of one extra checkpoint load")
    parser.add_argument("--year-offset", default="auto",
                        help="years added to model dates so pandas can hold them "
                             "(control runs use years like 0425); 'auto' shifts only "
                             "when needed, by whole centuries into 2000-2099")
    parser.add_argument("--list-modules", action="store_true",
                        help="print the modules that would be hooked, then exit")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from fme.core.cli import prepare_directory
    from fme.core.timing import GlobalTimer

    config, config_data, coupling, mode, run_from_config, module = load_config(args.config)
    print(f"config: {coupling} {mode}")

    if coupling == "coupled" and args.component is None:
        parser.error(
            "This is a coupled (SamudrACE) config: it holds an SFNO atmosphere and "
            "a Samudra ocean. Pass --component atmosphere or --component ocean."
        )
    if coupling != "coupled" and args.component is not None:
        print("note: --component is ignored for a single-network config")

    component = args.component or "atmosphere"
    arch = args.arch or COMPONENT_ARCH[component]
    spec = ARCHS[arch]
    pattern = args.module_pattern or spec["pattern"]
    template = args.filename_template or spec["template"]
    if args.module_types is None:
        types = spec["types"]
    elif len(args.module_types) == 1 and args.module_types[0].lower() == "any":
        types = None
    else:
        types = tuple(args.module_types)
    dtype = np.float16 if args.dtype == "float16" else np.float32
    print(f"component: {component if coupling == 'coupled' else 'single network'}  "
          f"arch: {arch}  types: {list(types) if types else 'any'}")

    if args.list_modules:
        full_stepper = config.load_stepper()
        stepper = component_stepper(full_stepper, coupling, component)
        network = network_of(stepper)
        hooked = select_modules(network, pattern, types)
        for step, (name, mod) in enumerate(hooked):
            print(f"step {step}: {name}  ({type(mod).__name__})")
        print(f"component timestep: {component_timestep(full_stepper, coupling, component)}")
        if args.write_grid:
            write_grid_file(
                component_dataset_info(full_stepper, coupling, component),
                args.write_grid,
            )
        return

    # The first timestamp. Inference configs can say; evaluator configs cannot.
    first_time = None
    initialization_time = None
    if args.start_time is not None:
        first_time = _as_cftime(args.start_time)
    elif mode == "inference":
        initialization_time, _ = module._get_initialization_time_and_timestep(config)
        initialization_time = _as_cftime(initialization_time)
    else:
        parser.error(
            "An evaluator config has no initial-condition block to read the start "
            "time from. Pass --start-time with the first timestamp you expect, or "
            "use the matching inference config."
        )

    captured: dict[int, list[np.ndarray]] = defaultdict(list)
    handles = []
    state: dict[str, object] = {}

    def make_hook(step):
        def hook(_module, _inputs, output):
            if len(captured[step]) < args.max_times:
                captured[step].append(to_numpy(output, dtype))
        return hook

    # fme builds its own stepper inside the run loop, so wrap the config's
    # loader to hook whatever it hands back.
    original_load_stepper = config.load_stepper

    def load_stepper_with_hooks():
        stepper = original_load_stepper()
        target = component_stepper(stepper, coupling, component)
        hooked = select_modules(network_of(target), pattern, types)
        for step, (name, mod) in enumerate(hooked):
            handles.append(mod.register_forward_hook(make_hook(step)))
        state["timestep"] = component_timestep(stepper, coupling, component)
        state["dataset_info"] = component_dataset_info(stepper, coupling, component)
        print(f"hooked {len(hooked)} modules: {', '.join(n for n, _ in hooked)}")
        print(f"component timestep: {state['timestep']}")
        return stepper

    config.load_stepper = load_stepper_with_hooks

    prepare_directory(config.experiment_dir, config_data)
    try:
        # The run functions expect an active GlobalTimer, which fme's own
        # main() sets up for them.
        with torch.no_grad(), GlobalTimer():
            run_from_config(config)
    finally:
        for handle in handles:
            handle.remove()
        config.load_stepper = original_load_stepper

    if not captured:
        raise RuntimeError(
            "No activations were captured. Run with --list-modules to check the "
            "module pattern matches the blocks you want."
        )

    step_delta = state.get("timestep")
    if step_delta is None:
        raise RuntimeError("The stepper was never loaded, so its timestep is unknown.")
    if first_time is None:
        first_time = initialization_time + pd.Timedelta(step_delta).to_pytimedelta()

    calendar = getattr(first_time, "calendar", None) or "standard"
    year_offset = (
        auto_year_offset(first_time.year)
        if args.year_offset == "auto"
        else int(args.year_offset)
    )

    n_captured = min(len(arrays) for arrays in captured.values())
    for step in list(captured):
        captured[step] = captured[step][:n_captured]
    times, original = model_times(first_time, step_delta, n_captured, year_offset)
    print(f"captured {n_captured} timesteps from {original[0]} to {original[-1]} "
          f"({calendar} calendar)")
    if year_offset:
        print(f"model years shifted by {year_offset:+d} for pandas; set "
              f"display_year_offset: {year_offset} in paths.json to show the originals")

    write_latents(captured, times, args.out, template, dtype,
                  model_time=original, year_offset=year_offset, calendar=calendar)

    if args.write_grid and state.get("dataset_info") is not None:
        write_grid_file(state["dataset_info"], args.write_grid)

    print(
        "\nNow point paths.json at this directory (latent_dir) and run "
        "`streamlit run app.py`."
    )


if __name__ == "__main__":
    main()
