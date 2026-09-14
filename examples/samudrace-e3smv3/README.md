# Exploring SamudrACE-E3SMv3 latents

This example runs the latent space visualiser on a real, published coupled
climate emulator, [SamudrACE-E3SMv3](https://huggingface.co/allenai/SamudrACE-E3SMv3):
a Samudra ocean U-Net coupled to a stochastic SFNO atmosphere, trained on the
E3SMv3 preindustrial control run. It goes from nothing to the running app, then
further, into analyses the app does not do.

- [What you need](#what-you-need)
- [1. Run the model and open the app](#1-run-the-model-and-open-the-app)
- [2. Explore in the app](#2-explore-in-the-app)
- [3. Run the deeper analyses](#3-run-the-deeper-analyses)
- [4. Change the experiment](#4-change-the-experiment)
- [5. Point it at another fme checkpoint](#5-point-it-at-another-fme-checkpoint)
- [Troubleshooting](#troubleshooting)
- [Reading the results honestly](#reading-the-results-honestly)

Files here:

| File | What it does |
| --- | --- |
| `run.sh` | download, short rollout, latent extraction for both components, reference fields, app |
| `run_seed.sh` | a second rollout with a different noise seed, for the spread analysis |
| `analysis.py` | seven analyses beyond the app, writing maps and `results.json` |
| `extract-config.yaml` | the fme inference config used for extraction (with why it differs from the published one) |
| `paths.json` | the app configuration for both components |

---

## What you need

- **[uv](https://docs.astral.sh/uv/)** — `run.sh` builds its own Python 3.12
  environment with `fme`, `torch` and the app's requirements.
- **About 9 GB of disk** for `run.sh` (2.5 GB download, 2 GB of latents, 2.8 GB
  of rollout output, a 1.6 GB environment, reference fields), and another 5 GB
  for each extra seed from `run_seed.sh`. The `run_*` directories hold the full
  prediction files; once the reference fields are built you can delete them,
  except that the spread analysis reads them.
- **Memory**: the app holds one timestep of one component at a time — 1.2 GB for
  the ocean, 0.8 GB for the atmosphere. `analysis.py` peaks at a few GB.
- **A GPU is optional.** On Apple silicon the rollout runs on the GPU through
  MPS (`FME_USE_MPS=1`, set by the scripts), and each extraction takes a few
  minutes on an M1 Max. On CUDA machines fme uses the GPU automatically. On CPU
  only, expect it to be several times slower but still workable for four steps.
- **Network access to huggingface.co** for the download.

## 1. Run the model and open the app

From the repository root:

```bash
examples/samudrace-e3smv3/run.sh
```

By default everything goes to `demo_data/samudrace_e3smv3/` (git ignores it).
Pass a directory as the first argument to put it elsewhere. Each stage checks
for its own output and skips itself if it is there, so re-running only opens
the app; delete a stage's output to redo it. Set `NO_APP=1` to stop before the
app starts.

What it does, stage by stage, and what appears:

| Stage | Output | Notes |
| --- | --- | --- |
| environment | `.venv/` | `fme` installed from `mahf708/ace` main |
| download | `hf/` | checkpoint (2.2 GB), 3 initial conditions, 1 year of forcing |
| ocean latents | `latents/ocean/`, `ocean_grid.npz`, `run_ocean/` | 9 U-Net levels × 4 five-day steps; grid and land mask come from the checkpoint |
| atmosphere latents | `latents/atmos/`, `atmos_grid.npz`, `run_atmosphere/` | 8 SFNO blocks × the first 4 six-hour steps, stored as float16 |
| reference fields | `reference/*.nc` | the rollout's own state, built from `run_*/…/autoregressive_predictions.nc` |
| app | — | `streamlit run app.py`, opened in your browser |

The extraction script prints what it found. For this checkpoint:

```text
component: ocean  arch: samudra  types: ['ConvNeXtBlock']
hooked 9 modules: module.layers.0, module.layers.2, ... module.layers.16
component timestep: 5 days 00:00:00
captured 4 timesteps from 0425-01-08T12:00:00 to 0425-01-23T12:00:00 (noleap calendar)
model years shifted by +1600 for pandas; set display_year_offset: 1600 in paths.json ...
```

To start the app again later without the script:

```bash
cd demo_data/samudrace_e3smv3
.venv/bin/streamlit run ../../app.py
```

Run it from the data directory: the app reads `paths.json` from the current
directory.

## 2. Explore in the app

Pick **SamudrACE-E3SMv3 ocean (Samudra)** or **SamudrACE-E3SMv3 atmosphere
(SFNO)** in the sidebar, choose a time (model year 0425), press **Load data**,
and walk the five steps. Some things worth trying:

**Ocean**

- *Irminger Sea, level 0* — latitude 55, longitude −35, radius 10. Step 3's top
  channels trace western boundary currents worldwide; Step 5's PC0 loads on the
  same fronts.
- *Antarctic Circumpolar Current* — latitude −55, longitude 60. Nearly the whole
  Southern Ocean band comes out similar.
- *Niño 3.4* — latitude 0, longitude −145, radius 8. Compare level 0 with level 4
  (the 11×22 bottleneck): the maps turn blocky because the level genuinely is
  that coarse.
- Move the **U-Net level** slider from 0 to 8 and watch channel widths change
  (280 → 520 → 280). A channel number at level 2 is a different feature from the
  same number at level 6; the app says so.
- In Step 2, try `ocean_sea_ice_fraction`, `ssh`, and `temperatureCoarsened_10`
  (the 410–530 m layer). Depth indices follow the interfaces
  0, 20, 30, 40, 50, 80, 110, 140, 170, 230, 410, 530, 1020, … m.

**Atmosphere**

- *North Atlantic storm track, block 7* — latitude 45, longitude −30. Step 4's
  cosine similarity is high almost everywhere; see
  [Reading the results honestly](#reading-the-results-honestly) for why, and run
  the centred version in `analysis.py`.
- *Southern Ocean storm track* — latitude −50, longitude 90.
- *Tibetan Plateau* — latitude 32, longitude 88.
- Compare block 0 with block 7 in Step 3's line plot. Over the North Atlantic,
  channel 321 is the most active channel at blocks 0, 4 and 7, which is what you
  would expect if blocks share a latent space; check whether that holds where you
  look.
- In Step 2, vertical levels run from the top: `T_0` is the stratosphere
  (≈224 K), `T_7` the lowest level (≈275 K); `U_2` sits at jet height.

Both components live on the same 180×360 grid, so the same coordinates work in
both; land cells are simply not selectable for the ocean.

## 3. Run the deeper analyses

```bash
cd demo_data/samudrace_e3smv3
../../examples/samudrace-e3smv3/run_seed.sh 1        # second rollout, seed 1 (for "spread")
.venv/bin/python ../../examples/samudrace-e3smv3/analysis.py . analysis_out
```

The third argument limits which analyses run, comma-separated, and results
accumulate in `analysis_out/results.json`:

```bash
.venv/bin/python ../../examples/samudrace-e3smv3/analysis.py . analysis_out cka,centring
```

| Name | Question | Output |
| --- | --- | --- |
| `probes` | How much of the input state, and of the one-step change, can a linear readout recover from each level/block? | `probes` in JSON |
| `cka` | How similar are the representations at different depths, and between ocean and atmosphere? | `cka`, `cross_cka` |
| `dimension` | How many directions does each level/block actually use? | `dimension` |
| `centring` | What does cosine similarity look like with the global mean vector removed? | 4 maps |
| `dictionary` | Which single channel best tracks a physical variable? | 12 maps |
| `regions` | Which parts of the globe resemble a chosen region? | 6 maps |
| `spread` | Where do two rollouts with different noise seeds diverge, in latents and in physical fields? | `spread` |

On an M1 Max, `probes` takes about a minute, `spread` a little over a minute,
and the whole set a few minutes. Each reads latents through the app's loaders,
so a component's four timesteps are loaded one at a time.

How each one is computed, so you can judge it:

- **probes** — ridge regression from standardised latents to physical fields at
  the input time (*state*) and to their change over one step (*tendency*). Targets
  are anomalies from each latitude's mean, so a readout gets no credit for knowing
  the equator is warm. Scored by **spatially blocked 5-fold cross-validation**
  (15°×15° blocks), with the ridge strength chosen by inner cross-validation. A
  harsher west-to-east hemisphere transfer score is stored as
  `hemisphere_state` / `hemisphere_tendency`. Four timesteps, every second grid
  point.
- **cka** — linear centred kernel alignment on 8,000 ocean points at the first
  timestep. Ocean and atmosphere are compared on the same points, from the forward
  passes that read the same initial condition.
- **dimension** — participation ratio `(Σλ)² / Σλ²` of the channel covariance
  over all valid points, and the number of components for 90% of the variance.
- **centring** / **regions** — cosine similarity to a region's *mean* latent
  vector (the app uses the first node in the region), after subtracting the
  global mean latent vector.
- **dictionary** — for each variable, the level/block and channel with the highest
  |correlation| to the variable's zonal anomaly; channel maps whose correlation is
  negative are sign-flipped for display, and both raw and anomaly correlations are
  recorded.
- **spread** — RMS difference between the seed-0 and seed-1 latents, divided by
  the seed-0 spatial standard deviation, per step and time; and the same measure
  on the predicted physical fields.

To change regions, variables or steps, edit the `COMPONENTS` dictionary at the
top of `analysis.py`; every analysis reads its choices from there.

## 4. Change the experiment

Edit `examples/samudrace-e3smv3/extract-config.yaml` (it is copied into the data
directory on each `run.sh`), then delete the outputs you want regenerated.

| To… | Change | Then delete |
| --- | --- | --- |
| roll out longer | `n_coupled_steps` (each is 5 days) and `--max-times` in `run.sh` | `latents/`, `run_*`, `reference/` |
| start from another initial condition | `start_indices.first` (0, 1 or 2 — model years 0425, 0426, 0427) | same |
| capture more timesteps | `--max-times` in `run.sh`; memory in the app is per timestep, disk grows linearly | `latents/` |
| look at fewer levels | `--module-pattern '^(module\.)?layers\.(0|8|16)$'` on the ocean extraction | `latents/ocean` |
| store less | `--dtype float16` on the ocean as well | `latents/ocean` |
| use a different noise sequence | `seed` | `latents/`, `run_*`, `reference/` |
| compare seeds | `run_seed.sh N`, then `SPREAD_SEED=N analysis.py … spread` | — |

Before any long run, check what will be hooked without doing any work:

```bash
.venv/bin/python ../../scripts/extract_latents_fme.py extract-config.yaml \
    --component atmosphere --list-modules
```

Two things to keep consistent:

- **Seeds.** The model is stochastic. Ocean and atmosphere are extracted in
  separate runs and the reference fields come from those runs, so all of them must
  share a seed. With the same seed, runs are bit-identical.
- **Year offset.** Latents and reference fields must use the same
  `--year-offset` (1600 here), and `display_year_offset` in `paths.json` must
  match it, or Step 2 will not find the times.

## 5. Point it at another fme checkpoint

The same pieces work for other `fme` models; what changes is the config:

1. Start from the inference config published with the checkpoint. Shorten
   `n_forward_steps` (single models) or `n_coupled_steps` (coupled), set a `seed`
   if the model is stochastic, and turn on `save_prediction_files`.
2. Run `--list-modules`. For a single-network model, omit `--component`. If
   nothing matches, the script lists the module names; pass `--module-pattern`
   (and `--module-types`) to pick blocks.
3. Extract with `--write-grid`, build references with
   `scripts/fme_predictions_to_reference.py` (for a single-network model, pass the
   run directory itself), and copy `paths.json`, adjusting labels, timesteps and
   the year offset the script reports (0 for real-calendar runs such as ERA5).

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `curl: (56) CONNECT tunnel failed, response 403` | huggingface.co is blocked on your network. Download `hf/` elsewhere and copy it in; `run.sh` skips files that exist. |
| `OutOfBoundsDatetime: ... 0425-01-03` | An old copy of the extraction script without calendar handling. Update the repository. |
| Step 1: *Could not find ... in the reference data* | The year offset of the reference file and `display_year_offset` disagree, or the chosen time is the first one and has no preceding state. |
| `No modules matched` | The checkpoint's module names differ; read the listed names and pass `--module-pattern`. |
| `--component is required` | The config is coupled (`n_coupled_steps`). Pass `ocean` or `atmosphere`. |
| MPS errors about unsupported operations | `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the scripts; if running by hand, set it yourself or unset `FME_USE_MPS` to use the CPU. |
| The app is slow switching levels | Each step loads one timestep of all levels (1.2 GB for the ocean). Extract fewer levels. |
| `fatal: not a git repository` in the run log | fme trying to record a commit hash from the data directory. Harmless. |

## Reading the results honestly

These are exploratory analyses of one short rollout from one initial condition.
Some care points specific to what this example found:

- **Un-centred cosine similarity misleads on SFNO latents.** The atmosphere's
  latent vectors share a large common component: at block 7 the median
  similarity to a North Atlantic region is 0.84, and every grid point is above
  0.5. Removing the global mean vector brings the median to −0.02 and reveals
  structure (other storm tracks). The app's Step 4 does not centre; Step 5's PCA
  does.
- **Coarse levels cannot hold fine structure.** Anything scored per grid point
  (probes, dictionary, similarity) is capped at the bottleneck by its 11×22
  resolution, not only by what the level encodes.
- **Spatial autocorrelation inflates random splits.** Probes use spatially
  blocked cross-validation for that reason. The ocean's readouts also transfer
  poorly from one hemisphere to the other, a sign the latent-to-physics mapping is
  basin-dependent; look at `hemisphere_*` before generalising.
- **Single channels are not variables.** The best single-channel anomaly
  correlations are moderate (|r| ≈ 0.4–0.6); information is spread across
  channels, which is what the probes measure.
- **The atmosphere latents show faint seams at 90° longitude intervals** — the
  largest east–west jumps in the raw block-0 latents fall exactly at 90°, 180°,
  270° and 360°, about 25% above typical neighbour differences, while the input
  fields have none there. The cause is not established here (it could be model
  structure or a numerical effect of running on MPS); check before interpreting
  features that line up with those meridians.
- **Short rollouts, one member.** Four steps from one initial condition show what
  the latents look like, not how stable any finding is. Longer rollouts, other
  initial conditions and more seeds are the obvious next steps (section 4).
