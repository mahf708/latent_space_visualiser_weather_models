#!/usr/bin/env python3
"""Extract latent activations from an `fme` (ACE) model run, for the visualiser.

The ``fme`` package (https://github.com/mahf708/ace) does not write internal
activations during inference, so this script attaches forward hooks to the
network's blocks, runs fme's own inference loop, and writes what the hooks
captured in the layout the app's structure modules expect:

* ``--arch sfno``    -> ``latent_grid_block_<block>_<year>_<month>.npz``,
                        read by ``ace_structure_1`` (ACE, ACE2, any SFNO).
* ``--arch samudra`` -> ``latent_level_<level>_<year>_<month>.npz``,
                        read by ``samudra_structure_1``.

Each file holds ``latent`` with dimensions ``(time, batch, channel, lat, lon)``
and a ``time`` array of ISO timestamps, which is exactly what those structures
read.

Requirements
------------
``torch`` and ``fme``, which the Streamlit app itself does not need and which
are therefore not in ``requirements.txt``::

    pip install fme

Usage
-----
Start from an inference config that already runs for your checkpoint (see
https://ai2-climate-emulator.readthedocs.io/en/latest/quickstart.html), then::

    # check which modules would be hooked before committing to a long run
    python scripts/extract_latents_fme.py config-inference.yaml --arch sfno --list-modules

    python scripts/extract_latents_fme.py config-inference.yaml \
        --arch sfno --out demo_data/ace2_era5/latent_data --max-times 8

Notes
-----
* Activations are written raw: nothing is normalised or rescaled. Magnitudes are
  comparable across steps only where the architecture shares one latent space
  (SFNO blocks do; U-Net levels do not, which is why the app marks Samudra
  ragged).
* Memory is the limit. One ACE2 block at 1 degree with 256 channels is about
  33 MB per timestep, times the number of blocks. Use ``--max-times`` (and
  ``--dtype float16``) to keep a run manageable.
* The inference run itself proceeds normally and still writes its usual output
  to the config's ``experiment_dir``.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import numpy as np

DEFAULT_PATTERNS = {
    # SFNO processor blocks: fme.ace.models.modulus.sfnonet / makani.sfnonet
    # each output is (batch, embed_dim, lat, lon)
    "sfno": r"^(module\.)?blocks\.\d+$",
    # Samudra U-Net: one activation per level in the layer list
    "samudra": r"^(module\.)?layers\.\d+$",
}

DEFAULT_TEMPLATES = {
    "sfno": "latent_grid_block_{step}_{year}_{month:02d}.npz",
    "samudra": "latent_level_{step}_{year}_{month:02d}.npz",
}


def load_config(yaml_path):
    """Load an fme inference YAML into an InferenceConfig."""
    import dacite
    from fme.ace.inference.inference import InferenceConfig
    from fme.core.cli import prepare_config

    config_data = prepare_config(yaml_path)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    return config, config_data


def select_modules(model, pattern):
    """(name, module) pairs whose qualified name matches ``pattern``.

    Sorted by the trailing integer so step 10 does not sort before step 2.
    """
    regex = re.compile(pattern)
    selected = [(name, mod) for name, mod in model.named_modules() if regex.match(name)]
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


def network_of(stepper):
    """The underlying network of an fme stepper."""
    modules = list(stepper.modules)
    if not modules:
        raise RuntimeError("The stepper contains no modules.")
    if len(modules) > 1:
        print(f"note: the stepper has {len(modules)} modules; hooking the first")
    return modules[0]


def write_latents(captured, times, out_dir, template, dtype):
    """Write one file per hooked step, split by month, as the loaders expect."""
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
            np.savez_compressed(
                path,
                latent=stacked[positions].astype(dtype),
                time=np.array([t.isoformat() for t in times[positions]]),
            )
            print(f"wrote {path}  shape={stacked[positions].shape}")


def _group_by_month(times):
    """[( (year, month), positions ), ...] preserving time order."""
    groups = defaultdict(list)
    for position, timestamp in enumerate(times):
        groups[(timestamp.year, timestamp.month)].append(position)
    return [(key, np.array(positions)) for key, positions in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="fme inference YAML config")
    parser.add_argument("--out", default="latent_data",
                        help="directory to write latent files into")
    parser.add_argument("--arch", choices=sorted(DEFAULT_PATTERNS), default="sfno",
                        help="preset of module names and output filenames")
    parser.add_argument("--module-pattern", default=None,
                        help="regex over model.named_modules() names, overriding --arch")
    parser.add_argument("--filename-template", default=None,
                        help="output filename template, overriding --arch")
    parser.add_argument("--max-times", type=int, default=8,
                        help="stop capturing after this many forward passes "
                             "(the inference run itself continues)")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="dtype to store activations in")
    parser.add_argument("--start-time", default=None,
                        help="ISO timestamp of the first captured step; by default "
                             "this is derived from the config's initial condition")
    parser.add_argument("--list-modules", action="store_true",
                        help="print the modules that would be hooked, then exit")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from fme.ace.inference.inference import (
        _get_initialization_time_and_timestep,
        run_inference_from_config,
    )
    from fme.core.cli import prepare_directory
    from fme.core.timing import GlobalTimer

    pattern = args.module_pattern or DEFAULT_PATTERNS[args.arch]
    template = args.filename_template or DEFAULT_TEMPLATES[args.arch]
    dtype = np.float16 if args.dtype == "float16" else np.float32

    config, config_data = load_config(args.config)

    if args.list_modules:
        stepper = config.load_stepper()
        for step, (name, module) in enumerate(select_modules(network_of(stepper), pattern)):
            print(f"step {step}: {name}  ({type(module).__name__})")
        return

    # Timestamps come from the config's initial condition. This loads the
    # checkpoint once on its own, before the hooks are installed below.
    if args.start_time is None:
        initialization_time, timestep = _get_initialization_time_and_timestep(config)
        first_time = pd.to_datetime(str(initialization_time)) + pd.Timedelta(timestep)
        step_delta = pd.Timedelta(timestep)
    else:
        _, timestep = _get_initialization_time_and_timestep(config)
        first_time = pd.Timestamp(args.start_time)
        step_delta = pd.Timedelta(timestep)

    captured: dict[int, list[np.ndarray]] = defaultdict(list)
    handles = []
    hooked_names: list[str] = []

    def make_hook(step):
        def hook(_module, _inputs, output):
            if len(captured[step]) < args.max_times:
                captured[step].append(to_numpy(output, dtype))
        return hook

    # fme builds its own stepper inside the inference loop, so wrap the config's
    # loader to hook whatever it hands back.
    original_load_stepper = config.load_stepper

    def load_stepper_with_hooks():
        stepper = original_load_stepper()
        for step, (name, module) in enumerate(select_modules(network_of(stepper), pattern)):
            handles.append(module.register_forward_hook(make_hook(step)))
            hooked_names.append(name)
        print(f"hooked {len(hooked_names)} modules: {', '.join(hooked_names)}")
        return stepper

    config.load_stepper = load_stepper_with_hooks

    prepare_directory(config.experiment_dir, config_data)
    try:
        # run_inference_from_config expects an active GlobalTimer, which
        # fme.ace.inference's own main() sets up for it.
        with torch.no_grad(), GlobalTimer():
            run_inference_from_config(config)
    finally:
        for handle in handles:
            handle.remove()
        config.load_stepper = original_load_stepper

    if not captured:
        raise RuntimeError(
            "No activations were captured. Run with --list-modules to check the "
            "module pattern matches the blocks you want."
        )

    n_captured = min(len(arrays) for arrays in captured.values())
    for step in list(captured):
        captured[step] = captured[step][:n_captured]
    times = [first_time + i * step_delta for i in range(n_captured)]
    print(f"captured {n_captured} timesteps from {times[0]} to {times[-1]}")

    write_latents(captured, times, args.out, template, dtype)
    print(
        "\nNow point paths.json at this directory (latent_dir) and run "
        "`streamlit run app.py`."
    )


if __name__ == "__main__":
    main()
