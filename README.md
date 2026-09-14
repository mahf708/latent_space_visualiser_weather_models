# AI Weather Model Latent Space Visualiser

## Overview

![Example latent activation map](docs/latent_channel_activation_map_read_me.png)

*Example latent activation map showing spatial structure of a selected channel from Graphcast small.*

This repository contains a Streamlit application for exploring the latent space of AI-based weather and climate models. The app enables interactive analysis of latent features, spatial patterns, and their relationship to physical variables. It supports several model families out of the box and is created in a way to allow for easy adaption.

| Model family | Latent layout | Structure module |
| --- | --- | --- |
| GraphCast | one vector per icosahedral mesh node, per processor step; all steps share one latent space | `graphcast_structure_1.py` |
| ACE / ACE2 (SFNO), and other [`fme`](https://github.com/mahf708/ace) spherical models | `(embed_dim, lat, lon)` per network block; all blocks share one latent space | `ace_structure_1.py` |
| Samudra (ocean U-Net), and other encoder/decoder models | one activation per U-Net level, each with its own resolution and channel width | `samudra_structure_1.py` |

The app itself is model agnostic. Whatever the model, latents are presented to
the analysis steps as `(n_steps, n_nodes, latent_dim)`, where a node is any
point with a latitude and longitude: a mesh node for GraphCast, a grid cell for
ACE and Samudra. Nodes with no data (land, for an ocean model) and channels that
do not exist at a step (the narrower levels of a U-Net) are carried as NaN and
excluded from every statistic.

***The aim of this tool is to be able to simply carry out first analysis of the latent space, and to inspire research in this area***. 

The workflow includes:

1. Selecting a model and forecast time
2. Selecting a geographic region
3. Extracting latent representations
4. Performing cosine similarity analysis
5. Performing principal component analysis (PCA)

This code accompanies the results presented in the associated conference paper: https://link.springer.com/chapter/10.1007/978-3-032-29915-4_10 (arxiv: https://arxiv.org/abs/2604.20467)

---
## Requirements

- Python >= 3.12
- pip (latest recommended)

## Installation

Clone the repository:

```bash
git clone https://github.com/ktempestuous/latent_space_visualiser_weather_models.git
cd latent_space_visualiser_weather_models
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick demo (no data download)

To see the whole workflow without downloading anything, generate small synthetic
datasets for all three model families and point the app at them:

```bash
python scripts/make_demo_data.py --write-paths
streamlit run app.py
```

That writes `./demo_data` (about 25 MB) and a `paths.json` pointing at it. Pick
a model in the sidebar and walk through steps 1-5. The fields are synthetic,
built from a handful of smooth patterns on the sphere so the analyses produce
recognisable structure; they are not model output and say nothing about any real
model. Use `--models ace samudra` to generate only some of them.

A worked walkthrough of the demo, including what to expect at each step and how
the three model families differ, is published here:
https://claude.ai/code/artifact/03975af3-d778-4994-a111-e0fa04bcb281

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser.

In the case you are running the app on Jupyterlab, e.g. on University servers, do the following, replacing the placeholders: 
```bash
streamlit run app.py --server.port 8501
```
On a new browser, open
```bash
https://jupyter.{}.uni-{}.de/user/{user.name}/proxy/8501/
```

---

## Configuration and Extending the Tool

If you want to modify parameters, add new models, or adapt the data loading, the following files are the main entry points:

### `app_config.py`
- Central configuration file
- Edit default parameters (`APP_DEFAULTS`)
- Add or modify model definitions (`_definitions`, which builds `MODELS`)
- Configure data paths via `paths.json`
- Dispatches to the registered latent structure; there is no if/elif chain to extend

### `latent_structures.py`
- The registry every structure module registers itself with
- Shared helpers: `NodeGeometry`, grid-to-node flattening, dimension reordering,
  NaN padding of ragged steps, filename scanning, time axis construction

### `graphcast_structure_1.py`, `ace_structure_1.py`, `samudra_structure_1.py`
- One module per model family: how its latent files are named, read and mapped
  onto coordinates
- Copy the closest one when adding a family

### `utils.py`
- Plotting, geometry, and helper functions
- `plot_global_overlay_only` draws a scatter for unstructured meshes and a
  pcolormesh when a `grid_shape` is given
- Latent array helpers (`step_view`, `nanmax_abs`, `scatter_to_nodes`,
  `cosine_similarity_to`) that operate on the finite subset of a NaN-padded array

### `app.py`
- Main Streamlit application
- Controls UI workflow and interaction logic
- Add new analysis steps or modify existing ones here

### `scripts/`
- `make_demo_data.py` writes synthetic demo data for every supported structure
- `extract_latents_fme.py` captures real activations from an `fme` inference run
  via forward hooks, in the layout the structure modules expect

### Adding a New Model

If the model fits an existing family, no code is needed:

1. Add a block to `paths.json` with its data locations
2. Add an entry to `_definitions` in `app_config.py`, pointing `latent.structure`
   at the right structure module
3. Set `filename_template` and `dims` if its files differ from the defaults

### Adding a New Model Family

If the latents are laid out differently, write a structure module:

1. Copy the closest of `graphcast_structure_1.py` (unstructured mesh),
   `ace_structure_1.py` (regular grid) or `samudra_structure_1.py` (multi
   resolution, ragged channels)
2. Implement four functions: available months, available times, load latent, node
   geometry. `latent_structures.py` has helpers for most of the work
3. Call `register_structure(LatentStructure(...))` at the bottom of the module,
   giving it a `step_name` (what one step is called in the UI), and setting
   `supports_translator` and `ragged_channels`
4. Import the module in `app_config.py` so it registers

`load_latent` must return `(latent, step_labels, time_index)` with `latent`
shaped `(n_steps, n_nodes, latent_dim)`; `node_geometry` must return a
`NodeGeometry` with one latitude and longitude per node, optionally a
`grid_shape` and a `valid` mask.

### Extracting Latents from an `fme` Model

`fme` does not write internal activations during inference, so
`scripts/extract_latents_fme.py` attaches forward hooks to the network's blocks
and runs fme's own inference loop:

```bash
# check which modules would be hooked before a long run
python scripts/extract_latents_fme.py config-inference.yaml --arch sfno --list-modules

python scripts/extract_latents_fme.py config-inference.yaml \
    --arch sfno --out my_data/ace2/latent_data --max-times 8
```

`--arch sfno` hooks the SFNO processor blocks and writes files
`ace_structure_1` reads; `--arch samudra` hooks the U-Net's ConvNeXt blocks
(one activation per level, skipping the pooling, upsampling and output layers)
and writes files `samudra_structure_1` reads. Anything else: pass
`--module-pattern` with a regex matched against `model.named_modules()`, and
`--module-types` to filter by class. This script needs `torch` and `fme`, which
the app itself does not, so they are not in `requirements.txt`.

The script reads whichever config you already run the model with and works out
which of fme's four entry points it belongs to: `n_forward_steps` means the
single-network path, `n_coupled_steps` the coupled one, and within each a
`loader` means an evaluator config while `initial_condition` + `forcing_loader`
means an inference config.

### Coupled models (SamudrACE)

A coupled checkpoint holds two networks — an SFNO atmosphere and a Samudra
ocean — so `--component` is required; the script refuses to guess, because
`CoupledStepper.modules` lists the atmosphere first and picking wrong would
silently give you the wrong model's latents.

#### Worked example: SamudrACE-E3SMv3

[`examples/samudrace-e3smv3/`](examples/samudrace-e3smv3) runs the whole thing
on the published [SamudrACE-E3SMv3](https://huggingface.co/allenai/SamudrACE-E3SMv3)
checkpoint — download, a short rollout, latent extraction for both components,
reference fields, and the app:

```bash
examples/samudrace-e3smv3/run.sh            # data goes to demo_data/samudrace_e3smv3
```

It downloads about 2.7&nbsp;GB and needs [uv](https://docs.astral.sh/uv/). On an
M1 Max the rollout runs on the GPU and both extractions finish in a few minutes.
Every stage skips itself when its output exists, so re-running just opens the
app.

What that checkpoint turns out to be, as reported by the extraction script:

| | Ocean (Samudra) | Atmosphere (SFNO) |
| --- | --- | --- |
| Hooked modules | 9 × `module.layers.N` (ConvNeXtBlock) | 8 × `module.conditional_model.blocks.N` |
| Channels | 280, 380, 480, 520, 520, 480, 380, 280, 280 | 384 at every block |
| Level grids | 180×360 down to 11×22 and back | 180×360 |
| Timestep | 5 days | 6 hours |
| Land mask | yes (69% of cells are ocean) | none |
| In the app, one timestep | 1.21&nbsp;GB | 0.80&nbsp;GB |

Its channel widths are not fme's defaults, and the atmosphere is the
*stochastic* `NoiseConditionedSFNO`, whose blocks sit one level down under
`conditional_model` — both of which the presets now handle.

Three things about real model runs that the example deals with, and that you will
meet with your own:

- **Model calendars.** The E3SMv3 control run is in model year 0425 on a no-leap
  calendar, which pandas cannot represent (it stops at 1677). The extraction
  script steps through time in the model's own calendar, then shifts years by
  whole centuries into range (0425 → 2025) and stores the original dates as
  `model_time`. Set `display_year_offset` in `paths.json` to the offset it
  reports and the app shows the model's own years again. Override with
  `--year-offset`.
- **Stochastic models.** Two unseeded rollouts differ, so latents from one run
  would not match reference fields from another. The example config sets
  `seed: 0`, and every run follows the same noise sequence.
- **Reference fields.** There is no ERA5 for a control run. The natural
  reference is the rollout's own state, so the config turns on
  `save_prediction_files` and `scripts/fme_predictions_to_reference.py`
  stitches the initial condition and predictions into the `(time, lat, lon)`
  file Step 2 reads, with the same year offset as the latents.

#### Doing it by hand

```bash
# always look first: prints the hooked modules and the component's timestep
python scripts/extract_latents_fme.py extract-config.yaml \
    --component ocean --list-modules

python scripts/extract_latents_fme.py extract-config.yaml \
    --component ocean --out latents/ocean --write-grid ocean_grid.npz --max-times 4

python scripts/fme_predictions_to_reference.py run/ocean \
    reference/ocean_reference.nc --year-offset 1600
```

`--write-grid` saves the component's latitudes, longitudes and land mask from
the checkpoint's own dataset info, so there is no separate grid file to find.
See [`examples/samudrace-e3smv3/paths.json`](examples/samudrace-e3smv3/paths.json)
for the matching configuration. The two components are separate entries because
they live on different grids with different step counts; switch between them in
the app's model selector.

If memory is tight, extract a subset of levels with a narrower
`--module-pattern` (for example `'^(module\.)?layers\.(0|8|16)$'`), and store the
atmosphere as `--dtype float16`.

#### A caveat for Step 4 on SFNO latents

On the SamudrACE atmosphere, cosine similarity against a North Atlantic node is
above about 0.6 almost everywhere on the globe. SFNO latent vectors share a large
common component (the most active channel there is strongly negative worldwide),
so un-centred cosine similarity mostly measures that shared offset rather than
regional structure. Step 4 does not currently centre the latents; bear this in
mind when reading its maps for SFNO models, and prefer Step 5's PCA, which
centres on the selected region.

### Changing Default Parameters

Default parameters (e.g. number of channels, processor steps, figure sizes) can be modified in:

- `APP_DEFAULTS` in `app_config.py`

Changes will automatically propagate through the app.

---

## Data Requirements

This application requires external data. A GraphCast sample is available at
https://syncandshare.lrz.de/getlink/fiKpbQtFvZChe1yfM7PqaX/demo_data (the latent
data is in a separate zip file which needs to be unzipped before use), and
`scripts/make_demo_data.py` generates synthetic data for every model family.

Required data, per model:

* **Latent data** (extracted from the model; see the extraction script above)
* **Reference data** - a gridded physical dataset such as ERA5, used for the
  maps in Step 2 and for relating latent structure to physical variables
* **Coordinates** - mesh node features (GraphCast) or a grid/ocean-mask file
  (ACE, Samudra). Grid-based models can instead store `lat`/`lon` in the latent
  files themselves, or fall back to a regular global grid of the latent shape
* **Translator matrices** (optional, and only for families that share one latent
  space across steps)

### Directory Configuration

All data paths are configured via a local configuration file.

1. Copy the example config:

```bash
cp paths.example.json paths.json
```

2. Edit `paths.json` to point to your local data directories.

Only the models present in `paths.json` are offered in the app, so you can
configure just the ones you have.

Example:

```json
{
  "graphcast_small": {
    "latent_dir": "/path/to/latent_data",
    "translator_dir": "/path/to/translators",
    "era5_basepath": "/path/to/era5",
    "graph_coords_filepath": "/path/to/mesh_nodes.npy"
  },
  "ace2_era5": {
    "latent_dir": "/path/to/ace_latents",
    "reference_basepath": "/path/to/ace_reference",
    "reference_filename": "ace2_era5_reference_{ym_str}.nc"
  },
  "samudra_ocean": {
    "latent_dir": "/path/to/samudra_latents",
    "reference_basepath": "/path/to/ocean_reference",
    "reference_filename": "samudra_reference.nc",
    "grid_coords_filepath": "/path/to/ocean_grid.npz"
  }
}
```

`reference_filename` may contain `{ym_str}` (`YYYYMM`), `{year}` and `{month}`;
with no placeholder it is treated as a single file or store covering the whole
record. Optional keys: `filename_template` and `dims` if your latent files
differ from the defaults below, `reference_label` for the name shown in the UI,
`timestep_hours` / `timestep_freq` for the spacing between steps, and `mask_key`
to name the ocean mask variable.

---

## Data Format

### Latent Data - GraphCast (`graphcast_structure_1`)

One file per processor step, per month:

```
latent_mesh_step_<step>_<year>_<month>.npz
```

Each file holds an array with shape:

```
(timesteps, nodes, batch, latent_dim)
```

### Latent Data - ACE / SFNO (`ace_structure_1`)

One file per network block, per month:

```
latent_grid_block_<block>_<year>_<month>.npz
```

Each file holds `latent` with dimensions given by `dims`, by default:

```
(time, batch, channel, lat, lon)
```

and may also hold `lat`, `lon` and `time` arrays, which are used in preference
to anything inferred. A single file holding every block is supported too: drop
`{step}` from `filename_template` and add `step` to `dims`. `.nc` files are read
through xarray, in which case `dims` can be omitted.

### Latent Data - Samudra (`samudra_structure_1`)

One file per U-Net level, per month:

```
latent_level_<level>_<year>_<month>.npz
```

Same dimensions as above. Levels may differ in both resolution and channel
width: coarse levels are resampled (nearest neighbour) onto the model's output
grid, and narrower levels are NaN-padded to the widest. Because channel *k* is
a different feature at each level, this family is marked ragged and the app says
so in Step 3.

---

### Reference Data

NetCDF (or zarr) files with:

* dimensions `time`, `lat`, `lon` (`latitude`/`longitude` also accepted)
* the timestep being analysed, and the one before it

Where `timestep_hours` is set (GraphCast and ACE both step 6-hourly) the
preceding state is taken at exactly that offset, crossing a month boundary into
the neighbouring file if needed. Where it is null (monthly ocean output) the
previous timestamp present in the data is used instead.

Default filename formats:

```
Graphcast_small_processed_input_<YYYYMM>.nc
ace2_era5_reference_<YYYYMM>.nc
samudra_reference.nc
```

---

### Coordinates

**Mesh nodes** (GraphCast) - a `.npy` file with at least three columns:

```
[cos(theta), cos(phi), sin(phi)]
```

**Grids** (ACE, Samudra) - a `.npz`, `.npy`, `.nc` or `.zarr` file holding `lat`
and `lon` (1-D vectors or 2-D curvilinear arrays), and optionally an ocean mask
named `wet`, `mask`, `ocean_mask` or `valid`. Nodes outside the mask are
excluded from region selection and from every analysis step. Grid nodes are
flattened row-major (latitude outer, longitude inner).

---

## Reproducibility

All results presented in the associated paper can be reproduced using this code and data.

Important notes:

* Use the same dataset versions as described in the paper
* The application workflow mirrors the analysis pipeline used in the study

---

## Usage Guide

### Step 1 — Select Parameters

* Choose model (the sidebar describes its latent layout)
* Select forecast time
* Optionally enable translator, where the model has one
* Select number of top latent channels

### Step 2 — Select Region

* Choose variable and level from the reference data
* Select geographic location
* Define analysis radius

### Step 3 — Extract Latents

* Loads latent representations for every step of the network
* Identifies top activated channels in the selected region
* Visualises global activations, as a mesh scatter or a filled grid depending on
  the model
* The step slider is labelled for the model: processor step, block, or U-Net
  level

### Step 4 — Cosine Similarity

* Compares selected region with global latent structure
* Uses only the channels that exist at the selected step

### Step 5 — PCA

* Fits PCA on the selected nodes that carry data at the selected step
* Projects components across all nodes with data
* Displays dominant latent features, reported with the model's own channel
  numbering

---

## Output

The application allows exporting results as a PDF report, including:

* Selected parameters
* Generated figures
* PCA summaries

---

## Notes

* Users are able to acquire the latent datasets independently, by extracting intermediate states from AI models; `scripts/extract_latents_fme.py` does this for models in the `fme` package
* Additional models can be incorporated by adding an entry to `app_config.py`; additional model *families* by creating a new structure module (e.g. `ace_structure_1.py`) and registering it
* Paths must be configured locally via `paths.json`
* Synthetic demo data for every supported family can be generated with `scripts/make_demo_data.py`

---

## Citation

If you use this code, please cite:

```
Tempest, K.I., Beylich, M., Craig, G.C. (2026). Mechanistic Interpretability Tool for AI Weather Models. In: Paszynski, M., Barnard, A.S., Zhang, Y.J. (eds) Computational Science – ICCS 2026 Workshops. ICCS 2026. Lecture Notes in Computer Science, vol 16788. Springer, Cham. https://doi.org/10.1007/978-3-032-29915-4_10
```

