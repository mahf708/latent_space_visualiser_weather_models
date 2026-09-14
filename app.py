import io
import time
from calendar import month_name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

from app_config import *
from utils import *

Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main h1 {
        margin-bottom: 0.2rem;
    }
    .main h2 {
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
    .main p {
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="AI weather model latent space visualiser", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0.2rem;'>
        AI weather model latent space visualiser
    </h1>
    <p style='text-align: center; font-size: 1.15rem; color: #666; margin-top: 0;'>
        Identifying features from location
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# THEME SELECTION
# ---------------------------------------------------------
st.session_state.setdefault("theme_mode", "light")

theme_mode = st.sidebar.radio(
    "Theme",
    options=["light", "dark"],
    index=0 if st.session_state["theme_mode"] == "light" else 1,
    key="theme_mode",
)

current_theme = THEMES[st.session_state["theme_mode"]]
apply_theme_css(current_theme)

# ---------------------------------------------------------
# Functions for saving results: 
# --------------------------------------------------------- 

def _collect_export_metadata():
    """Collect current app selections/options for report export."""
    md = {
        "selected_model": st.session_state.get("selected_model"),
        "model_label": model_cfg.get("label"),
        "latent_structure": model_cfg["latent"].get("structure"),
        "latent_label": model_cfg["latent"].get("label"),
        "selected_flt_time": str(st.session_state.get("selected_flt_time")),
        "year_flt": st.session_state.get("year_flt"),
        "month_flt": st.session_state.get("month_flt"),
        "day_flt": st.session_state.get("day_flt"),
        "hour_flt": st.session_state.get("hour_flt"),
        "use_translator": st.session_state.get("use_translator"),
        "selected_var": st.session_state.get("selected_var"),
        "selected_level_value": st.session_state.get("selected_level_value"),
        "overlay_lat": st.session_state.get("overlay_lat"),
        "overlay_lon": st.session_state.get("overlay_lon"),
        "overlay_radius": st.session_state.get("overlay_radius"),
        "N_top": st.session_state.get("N_top"),
        "selected_proc_step": st.session_state.get("selected_proc_step"),
        "n_selected_nodes": None if st.session_state.get("selected_nodes") is None else len(st.session_state.get("selected_nodes")),
        "step_name": st.session_state.get("step_name"),
        "n_pcs": st.session_state.get("n_pcs"),
    }
    st.session_state["export_metadata"] = md


def build_export_pdf_bytes(include_step5=False):
    """Create a PDF in memory containing metadata + saved figures."""
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # --- First page: metadata / selected options ---
        fig_meta, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")

        md = st.session_state.get("export_metadata", {})
        lines = [
            "AI weather model latent space visualiser export",
            "",
            f"Selected model: {md.get('selected_model')}",
            f"Model label: {md.get('model_label')}",
            f"Latent label: {md.get('latent_label')}",
            "",
            f"Selected flt time: {md.get('selected_flt_time')}",
            f"Year / month / day / hour (flt): {md.get('year_flt')} / {md.get('month_flt')} / {md.get('day_flt')} / {md.get('hour_flt')}",
            f"Use translator: {md.get('use_translator')}",
            f"Selected variable: {md.get('selected_var')}",
            f"Selected level: {md.get('selected_level_value')}",
            f"Overlay lat/lon: {md.get('overlay_lat')}, {md.get('overlay_lon')}",
            f"Overlay radius (deg): {md.get('overlay_radius')}",
            f"N_top: {md.get('N_top')}",
            f"Latent structure: {md.get('latent_structure')}",
            f"Selected {str(md.get('step_name') or 'processor step').lower()}: {md.get('selected_proc_step')}",
            f"Selected latent nodes: {md.get('n_selected_nodes')}",
            f"Number of PCs: {md.get('n_pcs')}",
        ]

        ax.text(
            0.03, 0.97,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        pdf.savefig(fig_meta, bbox_inches="tight")
        plt.close(fig_meta)

        figs = st.session_state.get("export_figures", {})

        ordered_keys = [
            "fig_era",
            "fig_res",
            "fig_line",
            "fig_hist_top",
            "fig_latents_0", "fig_latents_1", "fig_latents_2", "fig_latents_3", "fig_latents_4",
            "fig_latents_5", "fig_latents_6", "fig_latents_7", "fig_latents_8", "fig_latents_9",
            "fig_latents_10", "fig_latents_11", "fig_latents_12", "fig_latents_13", "fig_latents_14",
            "fig_latents_15", "fig_latents_16", "fig_latents_17",
            "fig_S_c_topN",
            "fig_S_c",
        ]

        if include_step5:
            ordered_keys.extend([f"fig_pca_{i}" for i in range(64)])

        for key in ordered_keys:
            fig = figs.get(key)
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")

        explained_df = st.session_state.get("df_explained_variance")
        if include_step5 and explained_df is not None:
            fig_explained_table = dataframe_to_figure(
                explained_df,
                title="Explained Variance Ratio for Each Principal Component",
                max_rows=30,
                fontsize=8,
            )
            pdf.savefig(fig_explained_table, bbox_inches="tight")
            plt.close(fig_explained_table)

        pca_df = st.session_state.get("df_pca_loadings")
        if include_step5 and pca_df is not None:
            fig_pca_table = dataframe_to_figure(
            pca_df,
            title="Top Contributing Latent Channels per Principal Component",
            max_rows=30,
            fontsize=7,
            )
            pdf.savefig(fig_pca_table, bbox_inches="tight")
            plt.close(fig_pca_table)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ---------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------
st.session_state.setdefault("reference_ready", False)
st.session_state.setdefault("location_ready", False)
st.session_state.setdefault("step3_done", False)
st.session_state.setdefault("step4_done", False)
st.session_state.setdefault("step5_done", False)
st.session_state.setdefault("step3_signature", None)
st.session_state.setdefault("step4_signature", None)
st.session_state.setdefault("step5_signature", None)
st.session_state.setdefault("N_top", 9)
st.session_state.setdefault("export_figures", {})
st.session_state.setdefault("export_metadata", {})

# ---------------------------------------------------------
# HEIRARCHY
# ---------------------------------------------------------

step1_done = st.session_state.get("reference_ready", False)
step2_done = st.session_state.get("location_ready", False)
step3_done = st.session_state.get("step3_done", False)
step4_done = st.session_state.get("step4_done", False)
step5_done = st.session_state.get("step5_done", False)

st.markdown("### Workflow")
st.markdown(f"""
- {'✅' if step1_done else '⬜'} **Step 1:** Select initial parameters
- {'✅' if step2_done else '⬜'} **Step 2:** Select location
- {'✅' if step3_done else '⬜'} **Step 3:** Extract latent data
- {'✅' if step4_done else '⬜'} **Step 4:** Cosine similarity analysis
- {'✅' if step5_done else '⬜'} **Step 5:** Principal component analysis
""")

# ---------------------------------------------------------
# STEP 1 — SETUP, MODEL AND TIMESTEP SELECTION
# ---------------------------------------------------------
section_card(
    "Step 1: Select initial parameters",
    """
Choose the model and forecast time you want to analyse.

- **Model selection** determines which latent data is available, and how it is laid out:
  an unstructured mesh (GraphCast), a latitude-longitude grid (ACE/SFNO), or the
  levels of a U-Net (Samudra).
- **Forecast time** defines the forecast latent timestep (flt) for your analysis.
- Optionally enable the **translator** to transform latent features, where the model
  has one.
- Select the number of **top activated channels** (`N_top`) to include in further analysis.

Once selected, the corresponding reference data for this time period will be loaded."""
)

# select model
st.sidebar.markdown("## Step 1 Settings")
selected_model = st.sidebar.selectbox(
    "Select model",
    list(MODELS.keys()),
    format_func=lambda key: MODELS[key]["label"],
)
model_cfg = MODELS[selected_model]
st.session_state["selected_model"] = selected_model
st.session_state["model_cfg"] = model_cfg

structure = structure_for(model_cfg)
step_name = structure.step_name
ref_label = reference_label(model_cfg)
st.session_state["step_name"] = step_name
if structure.description:
    st.sidebar.caption(structure.description)

# import variables and paths: 
batch_num = APP_DEFAULTS["batch_num"]
figsize = APP_DEFAULTS["figsize"]
figsize_hist = APP_DEFAULTS["figsize_hist"]
dpi = APP_DEFAULTS["dpi"]
col_1_width = APP_DEFAULTS["col_1_width"]
col_2_width = APP_DEFAULTS["col_2_width"]

# select timestep

try:
    available_months = get_available_months(model_cfg)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

if not available_months:
    st.error(
        f"No latent files found for {model_cfg['label']}. Check paths.json and make "
        "sure latent_dir contains files matching this model's filename template, "
        "e.g. latent_mesh_step_<step>_<year>_<month>.npz for GraphCast or "
        "latent_grid_block_<block>_<year>_<month>.npz for ACE."
    )
    st.stop()

years = sorted({y for y, m in available_months})
selected_year = st.sidebar.selectbox("Select Year", years)

months = sorted({m for y, m in available_months if y == selected_year})

if not months:
    st.error(f"No months found for selected year {selected_year}.")
    st.stop()

selected_month = st.sidebar.selectbox("Select Month", months, format_func=lambda m: month_name[m])

times = get_available_times(selected_year, selected_month, model_cfg)
if len(times) == 0:
    st.error(f"No forecast times found for {selected_year}-{selected_month:02d}.")
    st.stop()

formatted_times = [t.strftime("%Y-%m-%d %H UTC") for t in times]
selected_flt_time_label = st.sidebar.selectbox("Select latent forecast time", formatted_times)
selected_flt_time = times[formatted_times.index(selected_flt_time_label)]

year_flt = selected_flt_time.year
month_flt = selected_flt_time.month
day_flt = selected_flt_time.day
hour_flt = selected_flt_time.hour
selected_flt_timestep = formatted_times.index(selected_flt_time_label)

# --- Load the geometry the latents live on ---
# GraphCast returns mesh nodes; ACE and Samudra return their grid flattened to
# nodes, plus the grid shape (for map drawing) and a validity mask (ocean only).
try:
    geometry = get_node_geometry(model_cfg)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

node_lats = geometry.lat        # -90 -> 90
node_lons = geometry.lon        # -180 -> 180
node_grid_shape = geometry.grid_shape
node_valid = geometry.valid
st.session_state["node_lats"] = node_lats
st.session_state["node_lons"] = node_lons
st.session_state["node_grid_shape"] = node_grid_shape
st.session_state["node_valid"] = node_valid

# --- Use translator? ---
if supports_translator(model_cfg):
    use_translator = st.sidebar.checkbox(
        "Use translator",
        value=model_cfg["latent"].get("use_translator_default", True),
    )
else:
    use_translator = False

# Top N channels? 
N_top = st.sidebar.slider(
    "Number of top channels",
    min_value=3,
    max_value=18,
    step=3,
    key="N_top",
)

# --- session state setup ---cosine_sims_top15
if st.sidebar.button("Load data"):
    # Reset downstream steps when time/translator changes
    st.session_state["reference_ready"] = False
    st.session_state["location_ready"] = False
    st.session_state["step3_done"] = False
    st.session_state["step4_done"] = False
    st.session_state["step5_done"] = False
    st.session_state["step3_signature"] = None
    st.session_state["step4_signature"] = None
    st.session_state["step5_signature"] = None
    st.session_state["export_figures"] = {}
    st.session_state["export_metadata"] = {}

    # Persist selections
    st.session_state["selected_flt_timestep"] = selected_flt_timestep
    st.session_state["selected_flt_time"] = selected_flt_time
    st.session_state["year_flt"] = year_flt
    st.session_state["month_flt"] = month_flt
    st.session_state["day_flt"] = day_flt
    st.session_state["hour_flt"] = hour_flt
    st.session_state["use_translator"] = use_translator

    # Load the reference field at the forecast time and the step before it - cached
    st.write(f"Loading {ref_label} data...")
    try:
        ds_pair = load_reference_pair(flt_time=selected_flt_time, model_cfg=model_cfg)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.session_state["ds_pair"] = ds_pair
    st.session_state["reference_ready"] = True
    st.success(f"{ref_label} data loaded successfully.")
    st.info("Ready to select location.")
    st.divider()

# ---------------------------------------------------------
# STEP 2 — REFERENCE VARIABLE + LOCATION SELECTION
# ---------------------------------------------------------
if st.session_state.get("reference_ready", False):

    ds_pair = st.session_state["ds_pair"]
    selected_flt_time = st.session_state["selected_flt_time"]
    section_card(
    "Step 2: Select location",
    """
    Select the physical variable and region of interest.

    - The chosen variable helps relate **latent spatial structures** to physical quantities.
    - Two maps are shown:
        - **Fit**: the forecast initialisation state  
        - **Residual (flt − fit)**: the model’s predicted change over time  

    Use the selection controls below to define a geographic region.  
    Latent nodes within this region will be identified and used in subsequent analysis steps.
        """
    )

    # Reference grid coordinate names (lat/lon or latitude/longitude)
    ref_lat_name = next((n for n in ("lat", "latitude") if n in ds_pair.coords), "lat")
    ref_lon_name = next((n for n in ("lon", "longitude") if n in ds_pair.coords), "lon")

    # Variable selection
    all_vars = list(ds_pair.data_vars)
    st.sidebar.markdown("## Step 2 Settings")
    selected_var = st.sidebar.selectbox("Select variable", all_vars, key="selected_var")

    var_da = ds_pair[selected_var]

    # Time handling
    if "time" in var_da.dims:
        data_fit = var_da.isel(time=0)
        data_flt = var_da.isel(time=1) 
    else:
        data_fit = var_da
        data_flt = var_da

    # Level selection if exists
    level_dim = None
    for possible in ["plev", "level", "lev", "pressure_level"]:
        if possible in data_fit.dims:
            level_dim = possible
            break

    if level_dim is not None:
        levels = data_fit[level_dim].values
        selected_level_value = st.sidebar.selectbox(f"Select {level_dim}", levels, key="selected_level_value")
        level_idx = int(np.where(levels == selected_level_value)[0][0])
        data_fit = data_fit.isel({level_dim: level_idx})
        data_flt = data_flt.isel({level_dim: level_idx})
    else:
        selected_level_value = None

    # Time label: the reference pair is ordered [fit, flt]
    fit_time = pd.Timestamp(ds_pair["time"].values[0])

    # Overlay selection (lat/lon + radius)
    st.subheader("Overlay selection")
    overlay_lat = st.number_input(
        "Latitude (-90->90)",
        value=float(st.session_state.get("overlay_lat", 0.0)),
        min_value=-90.0,
        max_value=90.0,
        key="overlay_lat",
    )
    overlay_lon = st.number_input(
        "Longitude (-180->180)",
        value=float(st.session_state.get("overlay_lon", 0.0)),
        min_value=-180.0,
        max_value=180.0,
        key="overlay_lon",
    )
    overlay_radius = st.slider(
        "Overlay radius (degrees)",
        min_value=0.5,
        max_value=20.0,
        value=float(st.session_state.get("overlay_radius", 2.0)),
        key="overlay_radius",
    )

    circ_lats, circ_lons = make_circle_points(center_lat=overlay_lat, center_lon=overlay_lon, radius_deg=overlay_radius)
    st.session_state["circ_lats"] = circ_lats
    st.session_state["circ_lons"] = circ_lons

    # Residual = latent forcast timestep - fit.
    data_residual = (data_flt - data_fit).values
    st.session_state["data_fit"] = data_fit
    st.session_state["data_residual"] = data_residual

    # Plot
    col_plot, col_plot2 = st.columns([col_1_width, col_1_width])
    with col_plot:
        title = f"f(t_init) | {ref_label} ({selected_var}) @ forecast initialisation ({fit_time.strftime('%Y-%m-%d %H:%M UTC')})"
        if selected_level_value is not None:
            title += f", for {level_dim}={selected_level_value}"

        fig_era = plot_global_data_with_overlay(
            data=data_fit.values,
            lon=ds_pair[ref_lon_name].values,
            lat=ds_pair[ref_lat_name].values,
            title=title,
            cbar_label=selected_var,
            figsize=figsize,
            dpi=dpi,
            overlay_lats=circ_lats,
            overlay_lons=circ_lons,
        )
        st.session_state["export_figures"]["fig_era"] = fig_era
        st.pyplot(fig_era, clear_figure=False)

    with col_plot2:
        title_res = f"f(t) - f(t_init) | Residual of {selected_var}"
        if selected_level_value is not None:
            title_res += f", for {level_dim}={selected_level_value}"

        fig_res = plot_global_residual_with_overlay(
            data=data_residual,
            lon=ds_pair[ref_lon_name].values,
            lat=ds_pair[ref_lat_name].values,
            title=title_res,
            cmap="RdBu_r",
            cbar_label=selected_var,
            figsize=figsize,
            dpi=dpi,
            overlay_lats=circ_lats,
            overlay_lons=circ_lons,
        )
        st.session_state["export_figures"]["fig_res"] = fig_res
        st.pyplot(fig_res, clear_figure=False)

    st.caption("Circle shows the latent nodes that will be used in further calculations.")

    # store plots (to be put in pdf later) 
    _collect_export_metadata()

    if st.button("Use selected area", key="confirm_area_button"):
        st.session_state["location_ready"] = True
        st.session_state["step3_done"] = False
        st.session_state["step4_done"] = False
        st.session_state["step5_done"] = False
        st.session_state["step3_signature"] = None
        st.session_state["step4_signature"] = None
        st.session_state["step5_signature"] = None

    if st.session_state.get("location_ready", False):
        st.info("Area selected. Ready to extract latent data.")
    
    st.divider()

# ---------------------------------------------------------
# STEP 3 — SELECT MESH NODES + EXTRACT LATENTS
# ---------------------------------------------------------

def _compute_step3_results():
    """Compute Step 3 results and store them in session_state."""
    indices = select_nodes_within_radius(
        node_lats,
        node_lons,
        st.session_state["overlay_lat"],
        st.session_state["overlay_lon"],
        st.session_state["overlay_radius"] * 111,  # deg -> km approximation
        valid=node_valid,
    )
    st.session_state["selected_nodes"] = indices

    # Create selected coordinates df
    if len(indices) > 0:
        selected_lats = node_lats[indices]
        selected_lons = node_lons[indices]
        st.session_state["selected_coords_df"] = pd.DataFrame({"Latitude": selected_lats, "Longitude": selected_lons})
    else:
        st.session_state["selected_coords_df"] = None

    # Load latent (cached)
    st.write("Processing latent data...")
    start = time.time()
    try:
        latent_out, steps, time_flt_index = load_latent(flt_time=st.session_state["selected_flt_time"], model_cfg=st.session_state["model_cfg"] , batch_num=batch_num,use_translator=st.session_state["use_translator"])
    except (FileNotFoundError, ValueError, IndexError, KeyError) as error:
        st.session_state["step3_done"] = False
        st.error(f"Could not load latent data: {error}")
        return
    st.session_state["latent_t"] = latent_out  # (steps, nodes, latent dim)
    st.session_state["steps"] = steps
    st.session_state["time_flt_index"] = time_flt_index
    N_top = st.session_state["N_top"]
    st.success(f"Latent loaded in {time.time() - start:.3f} seconds")

    if latent_out.shape[1] != node_lats.size:
        st.session_state["step3_done"] = False
        st.error(
            f"The latent data has {latent_out.shape[1]} nodes but the model geometry "
            f"has {node_lats.size}. Check that the latent files and the coordinates "
            "file describe the same grid."
        )
        return

    if len(indices) == 0:
        st.session_state["latent_t_filtered"] = None
        st.session_state["top_channels_idx"] = None
        st.session_state["step3_done"] = False
        st.warning(
            "No latent nodes with data were found inside the selected area. "
            "Try a larger radius, or a location over ocean for an ocean model."
        )
        return

    latent_t_filtered = latent_out[:, indices, :]  # (steps, selected_nodes, latent_dim)
    st.session_state["latent_t_filtered"] = latent_t_filtered

    # find most activated channels within selected area, at the final step
    final_step = latent_t_filtered.shape[0] - 1
    abs_max_per_channel = nanmax_abs(latent_t_filtered[final_step, :, :], axis=0)  # (latent_dim,)
    # Channels absent at the final step (ragged architectures) must never win.
    ranked = np.argsort(np.where(np.isfinite(abs_max_per_channel), abs_max_per_channel, -np.inf))
    top_channels_idx = np.array(
        [ch for ch in ranked[::-1] if np.isfinite(abs_max_per_channel[ch])][:N_top],
        dtype=int,
    )

    if top_channels_idx.size == 0:
        st.session_state["top_channels_idx"] = None
        st.session_state["step3_done"] = False
        st.warning(
            f"No latent channels carry data at the final {step_name.lower()} for the "
            "selected nodes."
        )
        return

    st.session_state["top_channels_idx"] = top_channels_idx

    st.session_state["step3_done"] = True

def _render_step3_outputs():
    """Render Step 3 plots/results from session_state (no recompute)."""
    indices = st.session_state.get("selected_nodes", None)
    coords_df = st.session_state.get("selected_coords_df", None)
    latent_out = st.session_state.get("latent_t", None)
    latent_t_filtered = st.session_state.get("latent_t_filtered", None)
    top_channels_idx = st.session_state.get("top_channels_idx", None)
    N_top = st.session_state.get("N_top",None)

    if top_channels_idx is None or len(top_channels_idx) == 0:
        st.warning(
            "No top latent channels are available. "
            "Please run Step 3 again or adjust the selected region."
        )
        st.session_state["step3_done"] = False
        return

    if indices is None or len(indices) == 0:
        st.warning("No latent nodes selected in Step 3. Please select a circle region first.")
        return

    if coords_df is not None:
        st.markdown("### Coordinates of selected latent nodes")
        st.dataframe(coords_df)

    if latent_out is None:
        st.warning("Latent data is not loaded. Please rerun Step 3.")
        return

    steps = st.session_state.get("steps") or list(range(latent_out.shape[0]))
    n_steps = latent_out.shape[0]

    st.caption(
        f"Found {len(indices)} latent nodes. Use the side bar to select the "
        f"{step_name.lower()} for the histogram of channel activations below."
    )

    st.sidebar.markdown("## Step 3 Settings")
    # Models differ in how many steps they have, so a value carried over from a
    # previously selected model has to be clamped before the widget sees it.
    stored_step = int(st.session_state.get("selected_proc_step", min(10, n_steps - 1)))
    st.session_state["selected_proc_step"] = int(np.clip(stored_step, 0, n_steps - 1))

    if n_steps > 1:
        selected_proc_step = st.sidebar.slider(
            step_name,
            0,
            n_steps - 1,
            key="selected_proc_step",
            help=f"Index into the {n_steps} {step_name.lower()}s stored for this model "
                 f"(labels: {steps[0]}...{steps[-1]}).",
        )
    else:
        selected_proc_step = 0

    if has_ragged_channels(model_cfg):
        st.info(
            f"This model's {step_name.lower()}s have different channel widths and "
            "resolutions, so channel *k* is a different feature at each of them. "
            "Compare channels within one "
            f"{step_name.lower()}; read the cross-{step_name.lower()} line plot as "
            "activation strength only. Coarse levels are shown resampled onto the "
            "model's output grid, and gaps mean the channel does not exist there."
        )

    col_plot_1, col_plot_2 = st.columns([col_1_width, col_1_width])
    with col_plot_1:
        fig_line, ax_line = plt.subplots(figsize=figsize_hist, dpi=dpi)
        theme = THEMES[st.session_state.get("theme_mode", "light")]
        plot_colors = theme["plot_cat"]

        for i, ch in enumerate(top_channels_idx):
            top_vals = nanmax_abs(latent_t_filtered[:, :, ch], axis=1)
            ax_line.plot(
                np.arange(latent_t_filtered.shape[0]),
                top_vals,
                label=f"Channel {ch}",
                color=plot_colors[i % len(plot_colors)],
                linewidth=2,
            )       
        ax_line.set_xlabel(step_name)
        ax_line.set_ylabel("Max Latent Activation (selected nodes)")
        ax_line.set_title(f"Max value of top {N_top} latent channels across {step_name.lower()}s in selected region")
        ax_line.legend()
        st.pyplot(fig_line)

    with col_plot_2:
        fig_hist_top, ax_hist = plt.subplots(figsize=figsize_hist, dpi=dpi)
        theme = THEMES[st.session_state.get("theme_mode", "light")]
        plot_colors = theme["plot_cat"]

        for i, ch in enumerate(top_channels_idx[:N_top]):
            data_hist = latent_out[selected_proc_step, :, ch]
            data_hist = data_hist[np.isfinite(data_hist)]
            if data_hist.size == 0:
                continue
            ax_hist.hist(
                data_hist,
                density=True,
                alpha=0.55,
                color=plot_colors[i % len(plot_colors)],
                label=f"Channel {ch}",
            )
        ax_hist.set_xlabel("Latent value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"Histogram of top {N_top} latent channels at {step_name.lower()} {selected_proc_step} over globe")
        ax_hist.legend()
        plt.tight_layout()
        st.pyplot(fig_hist_top)

    st.session_state["export_figures"]["fig_line"] = fig_line
    st.session_state["export_figures"]["fig_hist_top"] = fig_hist_top

    st.caption(f"Note on line plot above: Shows max value of top {N_top} channels within the selected region, with the {N_top} channels having been selected at the final {step_name.lower()}.")

    # Tile plots
    st.markdown(f"### Global activations of top {N_top} latent channels")
    cols = st.columns(3)

    ds_pair = st.session_state["ds_pair"]
    data_fit = st.session_state["data_fit"]
    selected_var = st.session_state["selected_var"]  # owned by the widget key
    circ_lats = st.session_state["circ_lats"]
    circ_lons = st.session_state["circ_lons"]

    for i, ch in enumerate(top_channels_idx[:N_top]):
        with cols[i % 3]:
            fig_latents = plot_global_overlay_only(
                title=f"Latent Channel {ch} Activations",
                overlay_lats=node_lats,
                overlay_lons=node_lons,
                overlay_values=latent_out[selected_proc_step, :, ch],
                circle_lons=circ_lons,
                circle_lats=circ_lats,
                cmap="PuOr_r",
                grid_shape=node_grid_shape,
            )
            st.pyplot(fig_latents)
            st.session_state["export_figures"][f"fig_latents_{i}"] = fig_latents

if st.session_state.get("reference_ready", False) and st.session_state.get("location_ready", False):
    section_card(
        "Step 3: Extract latent data and first analysis",
        """
Extract latent representations for whole globe. Identify those in selected region and explore their strongest activations.

- Latent nodes within the selected area are identified.
- Latent features are loaded for all """ + step_name.lower() + """s.
- The most strongly activated latent channels are selected for further analysis.
- Global activation maps and summary plots are shown below.
        """
    )
    

    # Signature to avoid recompute unless inputs changed
    step3_signature = (
        st.session_state.get("year_flt"),
        st.session_state.get("month_flt"),
        st.session_state.get("day_flt"),
        st.session_state.get("hour_flt"),
        st.session_state.get("use_translator"),
        float(st.session_state.get("overlay_lat")),
        float(st.session_state.get("overlay_lon")),
        float(st.session_state.get("overlay_radius")),
        int(st.session_state.get("N_top")),
    )

    if st.button("Extract latent data", key="step3_button"):
        st.session_state["export_figures"] = {
            k: v for k, v in st.session_state.get("export_figures", {}).items()
            if k in ["fig_era", "fig_res"]
        }

        cached_step3_ok = (
            st.session_state.get("latent_t_filtered") is not None
            and st.session_state.get("top_channels_idx") is not None
            and st.session_state.get("selected_nodes") is not None
            and len(st.session_state.get("selected_nodes")) > 0
        )

        if (
            st.session_state.get("step3_signature") != step3_signature
            or not cached_step3_ok
        ):
            st.session_state["step3_signature"] = step3_signature
            _compute_step3_results()
        else:
            st.session_state["step3_done"] = True

    # Render persisted Step 3 outputs
    if st.session_state.get("step3_done", False):
        _render_step3_outputs()
        st.info("Ready to analyse latent space. Go ahead with either steps 4 or 5.")

    st.divider()
# ---------------------------------------------------------
# STEP 4 — COSINE SIMILARITY MAPS
# ---------------------------------------------------------
if st.session_state.get("step3_done", False):

    section_card(
        "Step 4: Cosine Similarity Analysis",
        """
Compare the selected region with the rest of the globe using cosine similarity.

- A representative latent vector from the selected region is chosen.
- Similarity is calculated against all other latent nodes.
- Results are shown using:
  - **Top N channels only**
  - **All latent channels**
        """
    )
    N_top = st.session_state["N_top"]

    def _compute_step4_results():
        top_channels_idx = st.session_state["top_channels_idx"]
        selected_proc_step = int(st.session_state["selected_proc_step"])
        latent_t = st.session_state["latent_t"]
        indices = st.session_state.get("selected_nodes")

        if latent_t is None or top_channels_idx is None or len(top_channels_idx) == 0:
            st.error("Step 3 results are missing. Please rerun Step 3.")
            st.session_state["step4_done"] = False
            return

        step_latents = latent_t[selected_proc_step]           # (nodes, latent_dim)
        node_has_data, channel_idx = step_view(latent_t, selected_proc_step)

        if channel_idx.size == 0 or not node_has_data.any():
            st.session_state["step4_done"] = False
            st.error(
                f"There is no latent data at {step_name.lower()} {selected_proc_step}. "
                f"Choose a different {step_name.lower()}."
            )
            return

        # Representative location: the first selected node that has data here
        candidates = [i for i in indices if node_has_data[i]]
        if not candidates:
            st.session_state["step4_done"] = False
            st.error(
                f"None of the selected nodes carry data at {step_name.lower()} "
                f"{selected_proc_step}."
            )
            return
        reference_node = int(candidates[0])
        st.session_state["reference_node"] = reference_node

        # Only channels that exist at this step can take part
        top_present = np.array(
            [ch for ch in top_channels_idx if ch in set(channel_idx.tolist())], dtype=int
        )
        if top_present.size == 0:
            st.session_state["step4_done"] = False
            st.error(
                f"None of the top {N_top} channels exist at {step_name.lower()} "
                f"{selected_proc_step}."
            )
            return

        n_nodes = step_latents.shape[0]

        # Cosine similarity: all channels available at this step
        latents_all = step_latents[np.ix_(node_has_data, channel_idx)]
        sims_all = cosine_similarity_to(step_latents[reference_node, channel_idx], latents_all)
        cosine_sims_all = scatter_to_nodes(sims_all, node_has_data, n_nodes)

        # Cosine similarity: top N channels only
        latents_topN = step_latents[np.ix_(node_has_data, top_present)]
        sims_topN = cosine_similarity_to(step_latents[reference_node, top_present], latents_topN)
        cosine_sims_topN = scatter_to_nodes(sims_topN, node_has_data, n_nodes)

        st.session_state["cosine_sims_all"] = cosine_sims_all
        st.session_state["cosine_sims_topN"] = cosine_sims_topN
        st.session_state["cosine_channels_used"] = int(channel_idx.size)
        st.session_state["cosine_top_used"] = int(top_present.size)
        st.session_state["step4_done"] = True

    step4_signature = (
        st.session_state.get("step3_signature"),
        int(st.session_state.get("selected_proc_step", 10)),
    )

    if st.button(
        "Do cosine similarity analysis",
        key="step4_button",
    ):
        if st.session_state.get("step4_signature") != step4_signature:
            st.session_state["step4_signature"] = step4_signature
            _compute_step4_results()
        else:
            st.session_state["step4_done"] = True

    if st.session_state.get("step4_done", False):
        ds_pair = st.session_state["ds_pair"]
        data_fit = st.session_state["data_fit"]
        selected_var = st.session_state["selected_var"]  # owned by the widget key
        circ_lats = st.session_state["circ_lats"]
        circ_lons = st.session_state["circ_lons"]
        cosine_sims_all = st.session_state["cosine_sims_all"]
        cosine_sims_topN = st.session_state["cosine_sims_topN"]

        col_plot_1, col_plot_2 = st.columns([col_1_width, col_1_width])
        with col_plot_1:
            fig_S_c_topN = plot_global_overlay_only(
                title=f"Cosine Similarity of top {st.session_state.get('cosine_top_used', N_top)} channels",
                overlay_lats=node_lats,
                overlay_lons=node_lons,
                overlay_values=cosine_sims_topN,
                circle_lons=circ_lons,
                circle_lats=circ_lats,
                cmap="PRGn",
                grid_shape=node_grid_shape,
            )
            st.pyplot(fig_S_c_topN)

        with col_plot_2:
            fig_S_c = plot_global_overlay_only(
                title="Cosine Similarity of all channels",
                overlay_lats=node_lats,
                overlay_lons=node_lons,
                overlay_values=cosine_sims_all,
                circle_lons=circ_lons,
                circle_lats=circ_lats,
                cmap="PRGn",
                grid_shape=node_grid_shape,
            )
            st.pyplot(fig_S_c)

        st.caption(
            f"Compared against latent node {st.session_state.get('reference_node')} "
            f"at {step_name.lower()} {int(st.session_state['selected_proc_step'])}, using "
            f"{st.session_state.get('cosine_channels_used')} channels."
        )
        # save plots to session state for later:    
        st.session_state["export_figures"]["fig_S_c_topN"] = fig_S_c_topN
        st.session_state["export_figures"]["fig_S_c"] = fig_S_c
        _collect_export_metadata()

        pdf_bytes_step4 = build_export_pdf_bytes(include_step5=False)

        st.download_button(
            label="Download PDF report",
            data=pdf_bytes_step4,
            file_name="latent_space_report.pdf",
            mime="application/pdf",
            key="download_step4_pdf",
            )

    st.divider()

# ---------------------------------------------------------
# STEP 5 — PCA
# ---------------------------------------------------------

if st.session_state.get("step3_done", False):
    section_card(
        "Step 5: Principal Component Analysis",
        """
Apply PCA to latent features in the selected region and project the learned components globally.

- PCA is fitted using the selected latent nodes.
- Principal components are then evaluated across all latent nodes with data.
- The plots show the spatial structure of each component.
- A summary table lists the strongest contributing latent channels for each component.
        """
    )

    # Controls
    st.sidebar.markdown("## Step 5 Settings")
    n_pcs = st.sidebar.number_input(
        "Number of PCs (PCA)",
        min_value=2,
        max_value=64,
        value=int(st.session_state.get("n_pcs", 8)),
        step=1,
        key="n_pcs",
    )

    def _compute_step5_results():
        """Fit PCA on selected-region latents (all latent channels), then transform all nodes."""
        selected_proc_step = int(st.session_state["selected_proc_step"])
        indices = st.session_state["selected_nodes"]

        if indices is None or len(indices) == 0:
            st.session_state["step5_done"] = False
            st.error("No latent nodes are selected. Please choose a region first and rerun Step 3.")
            return

        # full_latent: (nodes, latent_dim) at the selected step
        latent_t = st.session_state["latent_t"]
        full_latent = latent_t[selected_proc_step]

        # Only nodes and channels that carry data at this step take part: land
        # points and channels absent from this step are excluded, then written
        # back as NaN so the maps leave them blank.
        node_has_data, channel_idx = step_view(latent_t, selected_proc_step)
        st.session_state["pca_channel_idx"] = channel_idx

        if channel_idx.size == 0 or not node_has_data.any():
            st.session_state["step5_done"] = False
            st.error(
                f"There is no latent data at {step_name.lower()} {selected_proc_step}. "
                f"Choose a different {step_name.lower()}."
            )
            return

        fit_rows = np.array([i for i in indices if node_has_data[i]], dtype=int)
        if fit_rows.size == 0:
            st.session_state["step5_done"] = False
            st.error(
                f"None of the selected nodes carry data at {step_name.lower()} "
                f"{selected_proc_step}."
            )
            return

        # 1) Data to fit PCA on (selected region; every channel present here)
        X = full_latent[np.ix_(fit_rows, channel_idx)]  # (n_selected_nodes, n_channels)

        n_samples, n_features = X.shape
        requested_n_pcs = int(st.session_state["n_pcs"])
        max_valid_pcs = min(n_samples, n_features)

        if requested_n_pcs > max_valid_pcs:
            st.session_state["step5_done"] = False
            st.error(
                f"PCA cannot be computed with {requested_n_pcs} components for the current selection. "
                f"You selected {n_samples} latent nodes with data, so the maximum allowed number of PCs is {max_valid_pcs}. "
                f"Please reduce 'Number of PCs (PCA)' or select a larger region."
            )
            return

        # 2) Number of PCs
        pca = PCA(n_components=int(st.session_state["n_pcs"]))

        # 3) Data to transform later (all nodes with data; same channels)
        X_all = full_latent[np.ix_(np.flatnonzero(node_has_data), channel_idx)]

        # 4) Fit PCA
        pca.fit(X)

        # 5) Calculate explained variance ratios
        explained_var = pca.explained_variance_ratio_
        df_explained = pd.DataFrame({
            "Principal Component": [f"PC{i}" for i in range(len(explained_var))],
            "Explained Variance (%)": explained_var * 100,
        })

        df_explained["Explained Variance (%)"] = df_explained["Explained Variance (%)"].map("{:.2f}".format)
        st.session_state["df_explained_variance"] = df_explained.copy()

        # 6) Transform all nodes with data, then pad back to the full node list
        X_pca_valid = pca.transform(X_all)
        X_pca_all = scatter_to_nodes(
            X_pca_valid, node_has_data, full_latent.shape[0], n_cols=X_pca_valid.shape[1]
        )  # (n_nodes, n_pcs), NaN where there is no data

        # Persist
        st.session_state["pca_model"] = pca
        st.session_state["X_pca_all"] = X_pca_all
        st.session_state["step5_done"] = True

    # Signature to avoid recompute unless inputs changed
    step5_signature = (
        st.session_state.get("step3_signature"),
        int(st.session_state.get("selected_proc_step")),
        int(st.session_state.get("n_pcs")),
    )

    if st.button("Do PCA", key="step5_button"):
        if st.session_state.get("step5_signature") != step5_signature:
            st.session_state["step5_signature"] = step5_signature
            st.session_state["step5_done"] = False
            _compute_step5_results()
        else:
            st.session_state["step5_done"] = True

    # 7) Plot ALL PCs as tiles (3 per row)
    if st.session_state.get("step5_done", False) and ("X_pca_all" in st.session_state):
        df_explained = st.session_state.get("df_explained_variance")
        if df_explained is not None:
            st.markdown("### Explained variance ratio for each principal component")
            st.dataframe(df_explained, width="stretch")
        
        ds_pair = st.session_state["ds_pair"]
        data_fit = st.session_state["data_fit"]
        selected_var = st.session_state["selected_var"]
        circ_lats = st.session_state["circ_lats"]
        circ_lons = st.session_state["circ_lons"]

        X_pca_all = st.session_state["X_pca_all"]  # (n_nodes, n_pcs)
        n_pcs_fit = X_pca_all.shape[1]

        st.markdown("### Global activations of principal components")
        cols = st.columns(3)

        for pc_idx in range(n_pcs_fit):
            with cols[pc_idx % 3]:
                overlay_pc = X_pca_all[:, pc_idx]
                fig_pca = plot_global_overlay_only(
                    title=f"PC {pc_idx}",
                    overlay_lats=node_lats,
                    overlay_lons=node_lons,
                    overlay_values=overlay_pc,
                    circle_lons=circ_lons,
                    circle_lats=circ_lats,
                    cmap="BrBG_r",
                    grid_shape=node_grid_shape,
                )
                st.pyplot(fig_pca)
                st.session_state["export_figures"][f"fig_pca_{pc_idx}"] = fig_pca

        # PCA loading table: top contributing channels per PC
        pca = st.session_state["pca_model"]
        n_pcs_fit = pca.components_.shape[0]
        N_TOP = 6

        # PCA was fitted on the channels present at this step, so map the
        # component columns back to the model's own channel numbering.
        channel_idx = st.session_state.get("pca_channel_idx")
        if channel_idx is None:
            channel_idx = np.arange(pca.components_.shape[1])

        rows = []

        for pc_idx in range(n_pcs_fit):
            pc_loadings = pca.components_[pc_idx, :]  # (n_channels_present,)

            # Indices of top contributing channels by absolute loading
            top_indices = np.argsort(np.abs(pc_loadings))[::-1][:N_TOP]

            row = {
                "PC": f"PC{pc_idx}"
            }

            for rank, col in enumerate(top_indices, start=1):
                row[f"Ch {rank}"] = f"{int(channel_idx[col])} ({pc_loadings[col]:+.3f})"

            rows.append(row)

        df_pca_loadings = pd.DataFrame(rows)
        st.session_state["df_pca_loadings"] = df_pca_loadings
        st.markdown("### Top contributing latent channels per principal component")
        st.caption("Value in parenthesis is value of principle axis in feature space")
        st.dataframe(df_pca_loadings, width="stretch")

        # output plots created
        _collect_export_metadata()
        pdf_bytes_step5 = build_export_pdf_bytes(include_step5=True)

        st.download_button(
            label="Download PDF report",
            data=pdf_bytes_step5,
            file_name="latent_space_report.pdf",
            mime="application/pdf",
            key="download_step5_pdf",
             )
