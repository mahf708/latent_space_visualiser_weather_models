import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import streamlit as st
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

"""
utils.py

Utility for:

1 - make_circle_points - makes a circle of points on the sphere in degrees
2 - mesh_features_to_latlon - converts Graphcast mesh node features to coordinates on mesh node grid
3 - plot_global_data_with_overlay - plots 2D geospatial data (lat x lon) on a global Cartopy map with optional overlay points.
3b - plot_global_residual_with_overlay - for residual. Colourbar centred at zero.
4 - select_nodes_within_radius - select indices of latent nodes within a given radius from a centre point
5 - plot_global_overlay_only - plot data (e.g. latent channel value) on latent nodes,
    as a scatter for unstructured meshes or a pcolormesh for regular grids
6 - apply_translator - apply translator to latent features for an individual timestep and processor step, for all mesh nodes and latent channels
7 - dataframe_to_figure - render a pandas DataFrame as a matplotlib figure for PDF export
8 - apply_theme_css - app theme helper
9 - section_card - adds heirarchy to app
10 - step_view / nanmax_abs / scatter_to_nodes / cosine_similarity_to - latent
     array helpers that work on the finite subset of a NaN-padded latent array
"""

# ----------------------------------------------------
# --- 1. Makes a circle of points on the sphere in degrees
# ----------------------------------------------------

def make_circle_points(center_lat, center_lon, radius_deg, n=200):
    """Makes a circle of points on the sphere in degrees."""
    theta = np.linspace(0, 2 * np.pi, n)
    dlat = radius_deg * np.cos(theta)
    # degrees defined at the equator; scale longitude degrees by cos(lat)
    dlon = (radius_deg * np.sin(theta)) / np.cos(np.deg2rad(center_lat))
    return center_lat + dlat, center_lon + dlon

# ----------------------------------------------------
# --- 2. Convert features to (lat, lon)
# ----------------------------------------------------
def mesh_features_to_latlon(mesh_nodes_file):
    """
    Convert GraphCast mesh node features (cosθ, cosφ, sinφ) to latitude and longitude.
    Args:
        mesh_nodes_file: string of file name. Contains np.ndarray of shape (num_nodes, >=3)
    Returns:
        latitudes, longitudes: each np.ndarray of shape (num_nodes,)
    """
    mesh_nodes = np.load(mesh_nodes_file)
    
    cos_theta = mesh_nodes[:, 0]
    cos_phi = mesh_nodes[:, 1]
    sin_phi = mesh_nodes[:, 2]

    theta = np.arccos(cos_theta)           # polar angle [0, π]
    phi = np.arctan2(sin_phi, cos_phi)     # azimuth [-π, π]

    lat = 90.0 - np.degrees(theta)         # convert to latitude [-90, 90]
    lon = np.degrees(phi)                  # convert to longitude [-180, 180]
    return lat, lon

# --------------------------------------------------- 
# --- 3. Plot gloabl era5 data with option to overlay at select lat/lon coordinates
# ---------------------------------------------------

def plot_global_data_with_overlay(
    data,
    lon,
    lat,
    title="Map",
    cmap="viridis",
    cbar_label=None,
    figsize=(10, 5),
    overlay_lats=None,
    overlay_lons=None,
    dpi=100,
):
    """
    Plot 2D geospatial data (lat x lon) on a global Cartopy map
    with optional overlay points.

    Args:
        data (np.ndarray): 2D array (lat x lon)
        lon (np.ndarray): 1D longitude array
        lat (np.ndarray): 1D latitude array
        title (str): plot title
        cmap (str): colormap for base field
        cbar_label (str): label for colorbar
        figsize (tuple): figure size
        overlay_lats (np.ndarray): optional latitudes for overlay points
        overlay_lons (np.ndarray): optional longitudes for overlay points
        dpi (int): figure resolution
    """
    # Create 2D lon/lat grid
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Create figure with Cartopy projection
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi,
    )

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Base field using coordinate-aware plotting
    img = ax.pcolormesh(
        lon2d,
        lat2d,
        data,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        edgecolors=None,
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )

    # # Gridlines
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     draw_labels=True,
    #     linewidth=0.5,
    #     color="white",
    #     alpha=1,
    #     linestyle="--",
    #     xlocs=np.arange(-180, 181, 30),
    #     ylocs=np.arange(-90, 91, 30),
    # )
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlabel_style = {"size": 10}
    # gl.ylabel_style = {"size": 10}

    # Add axis ticks and labels
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=10)

    cbar = fig.colorbar(img, ax=ax, orientation="vertical", fraction=0.03, pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=12, pad=4)
    for tick in cbar.ax.get_yticklabels():
            tick.set_rotation(30)
            tick.set_ha("left")
            tick.set_va("center")
    cbar.locator = mticker.MaxNLocator(4)
    cbar.update_ticks()

    # Overlay points, no label
    if overlay_lats is not None and overlay_lons is not None:

    # outer white halo
        ax.plot(
            overlay_lons,
            overlay_lats,
            color="white",
            linewidth=2,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        # inner black line
        ax.plot(
            overlay_lons,
            overlay_lats,
            color="black",
            linewidth=1.25,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    ax.set_title(title)
    return fig


# --------------------------------------------------- 
# --- 3b. Plot gloabl era5 data with option to overlay at select lat/lon coordinates
# ---------------------------------------------------

def plot_global_residual_with_overlay(
    data,
    lon,
    lat,
    title="Map",
    cmap="viridis",
    cbar_label=None,
    figsize=(10, 5),
    overlay_lats=None,
    overlay_lons=None,
    dpi=100,
):
    """
    Plot 2D geospatial data (lat x lon) on a global Cartopy map
    with optional overlay points.

    Args:
        data (np.ndarray): 2D array (lat x lon)
        lon (np.ndarray): 1D longitude array
        lat (np.ndarray): 1D latitude array
        title (str): plot title
        cmap (str): colormap for base field
        cbar_label (str): label for colorbar
        figsize (tuple): figure size
        overlay_lats (np.ndarray): optional latitudes for overlay points
        overlay_lons (np.ndarray): optional longitudes for overlay points
        dpi (int): figure resolution
    """
    # Create 2D lon/lat grid
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Create figure with Cartopy projection
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi,
    )

    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    vmax = np.max(np.abs(data))

    if vmax == 0 or not np.isfinite(vmax):
        norm = None
        vmin, vmax = -1, 1
    else:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        vmin, vmax = -vmax, vmax

    # Base field using coordinate-aware plotting
    img = ax.pcolormesh(
        lon2d,
        lat2d,
        data,
        cmap=cmap,
        norm = norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        edgecolors=None,
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )

    # # Gridlines
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     draw_labels=True,
    #     linewidth=0.5,
    #     color="white",
    #     alpha=1,
    #     linestyle="--",
    #     xlocs=np.arange(-180, 181, 30),
    #     ylocs=np.arange(-90, 91, 30),
    # )
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlabel_style = {"size": 10}
    # gl.ylabel_style = {"size": 10}

    # Add axis ticks and labels
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=12, pad=4)
    for tick in cbar.ax.get_yticklabels():
            tick.set_rotation(30)
            tick.set_ha("left")
            tick.set_va("center")
    cbar.set_ticks([-vmax, -vmax/2, 0, vmax/2, vmax])


    # Overlay points, no label
    if overlay_lats is not None and overlay_lons is not None:

    # outer white halo
        ax.plot(
            overlay_lons,
            overlay_lats,
            color="white",
            linewidth=2,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        # inner black line
        ax.plot(
            overlay_lons,
            overlay_lats,
            color="black",
            linewidth=1.5,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    ax.set_title(title)
    return fig

# ---------------------------------------------------
# --- 4. Return indices around centre point for defined radius
# ---------------------------------------------------

def select_nodes_within_radius(latitudes: np.ndarray, longitudes: np.ndarray,
                               center_lat: float, center_lon: float,
                               radius_km: float, valid: np.ndarray = None):
    """
    Select indices of latent nodes within a given radius from a center point.

    Args:
        latitudes: np.ndarray of shape (num_nodes,)
        longitudes: np.ndarray of shape (num_nodes,)
        center_lat: latitude of center point in degrees
        center_lon: longitude of center point in degrees
        radius_km: radius in kilometers
        valid: optional boolean mask of shape (num_nodes,). Nodes that are
            False carry no latent data (e.g. land points of an ocean model)
            and are never selected.

    Returns:
        indices: np.ndarray of selected indices
    """
    # Earth's radius in km
    R = 6371.0

    # Convert degrees to radians
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    center_lat_rad = np.radians(center_lat)
    center_lon_rad = np.radians(center_lon)

    # Haversine formula
    dlat = lat_rad - center_lat_rad
    dlon = lon_rad - center_lon_rad
    a = np.sin(dlat/2)**2 + np.cos(center_lat_rad) * np.cos(lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c  # distance in km

    # Select indices within radius
    inside = distances <= radius_km
    if valid is not None:
        inside &= np.asarray(valid, dtype=bool)
    indices = np.where(inside)[0]
    return indices

# ---------------------------------------------------
# --- 5. Plot data (e.g. latent channel value) on mesh nodes
# ---------------------------------------------------
def plot_global_overlay_only(
    title="",
    figsize=(10, 5),
    overlay_lats=None,
    overlay_lons=None,
    overlay_values=None,
    circle_lons=None,
    circle_lats=None,
    dpi=100,
    cmap="PRGn",
    grid_shape=None,
):
    """
    Plot latent node values on a global Cartopy map without a background field.
    Colors are fully opaque and zero-centered.

    Nodes may be an unstructured mesh (GraphCast) or a flattened regular grid
    (ACE, Samudra). When ``grid_shape`` is given the values are drawn with
    pcolormesh, which fills the map correctly and is much faster than scattering
    one marker per grid cell; otherwise they are drawn as a scatter of nodes.

    Non-finite values are left blank. That is how the app shows nodes with no
    data: land points for an ocean model, and channels that do not exist at the
    selected step of a ragged (U-Net) architecture.

    Args:
        overlay_lats (np.ndarray): latitudes of the nodes
        overlay_lons (np.ndarray): longitudes of the nodes
        overlay_values (np.ndarray): values to plot at the nodes
        circle_lons (np.ndarray): optional circle longitude coordinates
        circle_lats (np.ndarray): optional circle latitude coordinates
        grid_shape (tuple): optional (n_lat, n_lon) if the nodes are a
            row-major flattened regular grid
    """

    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi,
    )

    ax.set_global()
    ax.coastlines(linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=6)

    # Gridlines
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     draw_labels=True,
    #     linewidth=0.5,
    #     color="white",
    #     linestyle="--",
    #     xlocs=np.arange(-180, 181, 30),
    #     ylocs=np.arange(-90, 91, 30),
    # )
    # gl.top_labels = False
    # gl.right_labels = False
        # Add axis ticks and labels
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=10)

    # --- Overlay only ---
    if overlay_lats is not None and overlay_lons is not None and overlay_values is not None:
        overlay_lats = np.asarray(overlay_lats, dtype=float).ravel()
        overlay_lons = np.asarray(overlay_lons, dtype=float).ravel()
        overlay_values = np.asarray(overlay_values, dtype=float).ravel()

        finite = np.isfinite(overlay_values)
        cmap = plt.get_cmap(cmap)

        if not finite.any():
            ax.set_title(title)
            ax.text(
                0.5, 0.5, "no data at these nodes",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
            )
            return fig

        max_abs = float(np.max(np.abs(overlay_values[finite])))
        if max_abs == 0.0 or not np.isfinite(max_abs):
            # TwoSlopeNorm needs vmin < vcenter < vmax; a constant-zero field
            # would otherwise raise.
            max_abs = 1.0

        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        if grid_shape is not None:
            # Regular grid: draw filled cells. Longitudes are wrapped into
            # -180 -> 180, which can leave a row starting mid-way round the
            # globe, so roll each row back into increasing order first.
            n_lat, n_lon = grid_shape
            lat2d = overlay_lats.reshape(n_lat, n_lon)
            lon2d = overlay_lons.reshape(n_lat, n_lon)
            values2d = overlay_values.reshape(n_lat, n_lon)

            shift = int(np.argmin(lon2d[0]))
            if shift:
                lat2d = np.roll(lat2d, -shift, axis=1)
                lon2d = np.roll(lon2d, -shift, axis=1)
                values2d = np.roll(values2d, -shift, axis=1)

            ax.pcolormesh(
                lon2d,
                lat2d,
                np.ma.masked_invalid(values2d),
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
                shading="nearest",
                rasterized=True,
                zorder=3,
            )
        else:
            colors = cmap(norm(overlay_values[finite]))
            colors[:, -1] = 1.0  # fully opaque

            ax.scatter(
                overlay_lons[finite],
                overlay_lats[finite],
                s=10,
                c=colors,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=12, pad=4)
        for tick in cbar.ax.get_yticklabels():
            tick.set_rotation(30)
            tick.set_ha("left")
            tick.set_va("center")
        cbar.set_ticks([-max_abs, -max_abs/2, 0, max_abs/2, max_abs])

    # --- Circle overlay ---
    if circle_lats is not None and circle_lons is not None:
        circle_lons = np.where(circle_lons > 180, circle_lons - 360, circle_lons)

        # outer white halo
        ax.plot(
            circle_lons,
            circle_lats,
            color="white",
            linewidth=2,
            alpha=1.0,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

        # inner black line
        ax.plot(
            circle_lons,
            circle_lats,
            color="black",
            linewidth=1.5,
            alpha=1.0,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )

    ax.set_title(title)

    return fig


# ---------------------------------------------------
# --- 6. Apply translator to latent features for an individual timestep and processor step, for all mesh nodes and latent channels
# ---------------------------------------------------

def apply_translator(latent_t, translator_file):
    """
    Apply translator weights and bias to latent tensor.

    Parameters
    ----------
    latent_t : np.ndarray
        Shape (nodes, batch_num, latent_dim), e.g., latent_array[selected_timestep,:,0,:]
    translator_file : str
        Path to .npz file containing W and b
        W shape: [latent_dim, latent_dim], b shape: [latent_dim]

    Returns
    -------
    np.ndarray
        Transformed latent tensor, same shape as latent_t: (nodes, batch_num, latent_dim)
    """
    npz = np.load(translator_file)
    W = np.array(npz["W"])
    b = np.array(npz["b"])

    # Apply linear transformation to each node and batch
    # latent_t: (nodes, latent_dim)
    # W: (latent_dim, latent_dim)
    # Result: (nodes, latent_dim)
    transformed = np.einsum("nd,df->nf", latent_t, W) + b

    return transformed

# ---------------------------------------------------
# --- 7. Render a pandas DataFrame as a matplotlib figure for PDF export
# ---------------------------------------------------

def dataframe_to_figure(df, title="", max_rows=30, fontsize=8):
    """Render a pandas DataFrame as a matplotlib figure for PDF export."""
    theme = THEMES.get(st.session_state.get("theme_mode", "light"), THEMES["light"])

    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(8.27, 2.0))
        ax.axis("off")
        ax.text(
            0.02, 0.8,
            f"{title}\n\nNo data available.",
            fontsize=11,
            va="top",
            color=theme["text"],
        )
        fig.patch.set_facecolor(theme["background"])
        return fig

    df_show = df.head(max_rows).copy()

    nrows, ncols = df_show.shape
    fig_height = min(11.0, 1.2 + 0.35 * (nrows + 1))
    fig_width = min(11.5, max(8.0, 1.4 * ncols))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(theme["background"])
    ax.set_facecolor(theme["background"])
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, pad=12, color=theme["text"])

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(theme["border"])
        if row == 0:
            cell.set_facecolor(theme["secondary_background"])
            cell.set_text_props(color=theme["text"], weight="bold")
        else:
            cell.set_facecolor(theme["background"])
            cell.set_text_props(color=theme["text"])

    plt.tight_layout()
    return fig

# ---------------------------------------------------
# --- 8. App theme helper
# ---------------------------------------------------

THEMES = {
    "light": {
        "background": "#FBF8F1",
        "secondary_background": "#F3EEE3",
        "sidebar_background": "#F6F1E6",
        "text": "#1C241D",
        "muted_text": "#5E6A5F",
        "primary": "#5E7F65",
        "primary_text": "#F8F6EF",
        "border": "#DDD5C7",
        "code_bg": "#ECE7DC",
        "font_body": "Inter, sans-serif",
        "font_heading": "Inter, sans-serif",
        "font_code": "JetBrains Mono, monospace",
        "plot_seq": ["#F8F6EF", "#E7EFD9", "#C9D8B6", "#8FAA7A", "#5E7F65"],
        "plot_cat": ["#5E7F65", "#7FA37F", "#A3B18A", "#D9CBB3", "#8C6F5A"],
        "plot_div": "BrBG",
        "checkbox_tick": "#1C241D",
    },
    "dark": {
        "background": "#102019",
        "secondary_background": "#1A2B22",
        "sidebar_background": "#0D1813",
        "text": "#F3EBDD",
        "muted_text": "#CDBFA8",
        "primary": "#E6D8BE",
        "primary_text": "#102019",
        "border": "#314238",
        "code_bg": "#16241D",
        "font_body": "Inter, sans-serif",
        "font_heading": "Inter, sans-serif",
        "font_code": "JetBrains Mono, monospace",
        "plot_seq": ["#16241D", "#35523A", "#5E7F65", "#8FAA7A", "#E6D8BE"],
        "plot_cat": ["#E6D8BE", "#A3B18A", "#7FA37F", "#5E7F65", "#8C6F5A"],
        "plot_div": "BrBG",
        "checkbox_tick": "#102019",
    },
}

def apply_theme_css(theme: dict):
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background-color: {theme["background"]};
            color: {theme["text"]};
            font-family: {theme["font_body"]};
        }}

        /* Main content area */
        [data-testid="stAppViewContainer"] > .main {{
            background-color: {theme["background"]};
        }}

        /* Remove the dark top header/banner area */
        [data-testid="stHeader"] {{
            background: {theme["background"]};
            border-bottom: 1px solid {theme["border"]};
        }}

        /* Toolbar area near the top */
        [data-testid="stToolbar"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background-color: {theme["sidebar_background"]};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {theme["text"]};
            font-family: {theme["font_heading"]};
        }}

        p, label, .stMarkdown, .stText {{
            color: {theme["text"]};
            letter-spacing: 0.01em;
        }}

        code, pre {{
            font-family: {theme["font_code"]} !important;
            background-color: {theme["code_bg"]} !important;
            color: {theme["text"]} !important;
            border-radius: 0.4rem;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background-color: {theme["primary"]} !important;
            color: {theme["primary_text"]} !important;
            border: 1px solid {theme["primary"]} !important;
            border-radius: 0.5rem;
        }}

        /* Force ALL nested text elements */
        .stButton > button *,
        .stDownloadButton > button * {{
            color: {theme["primary_text"]} !important;
            fill: {theme["primary_text"]} !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            filter: brightness(0.95);
        }}

        /* Inputs / selects */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {{
            background-color: {theme["secondary_background"]} !important;
            color: {theme["text"]} !important;
            border: 1px solid {theme["border"]} !important;
        }}

        /* Text inside select boxes */
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: {theme["text"]} !important;
        }}

        input, textarea {{
            color: {theme["text"]} !important;
            background-color: {theme["secondary_background"]} !important;
        }}

        /* Dropdown menu popover */
        [role="listbox"] {{
            background-color: {theme["secondary_background"]} !important;
            border: 1px solid {theme["border"]} !important;
        }}

        [role="option"] {{
            background-color: {theme["secondary_background"]} !important;
            color: {theme["text"]} !important;
        }}

        [role="option"]:hover {{
            background-color: {theme["background"]} !important;
        }}

        .stSlider, .stNumberInput, .stTextInput, .stSelectbox, .stMultiSelect {{
            color: {theme["text"]};
        }}

        /* Checkbox / radio clarity */
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {{
            color: {theme["text"]} !important;
        }}

        /* Checkbox box */
        [data-testid="stCheckbox"] div[role="checkbox"] {{
            background-color: {theme["secondary_background"]} !important;
            border: 1px solid {theme["border"]} !important;
        }}

        /* Checked state */
        [data-testid="stCheckbox"] div[aria-checked="true"] {{
            background-color: {theme["primary"]} !important;
            border: 1px solid {theme["primary"]} !important;
        }}

        /* The tick (SVG) */
        [data-testid="stCheckbox"] svg {{
            stroke: {theme["checkbox_tick"]} !important;
            fill: {theme["checkbox_tick"]} !important;
        }}

        [data-testid="stCheckbox"] input,
        [data-testid="stRadio"] input {{
            accent-color: {theme["primary"]};
        }}

        /* Dataframe / table styling */
        .stDataFrame, .stTable {{
            border: 1px solid {theme["border"]};
            border-radius: 0.5rem;
            background-color: {theme["background"]};
        }}

        [data-testid="stDataFrame"] {{
            background-color: {theme["background"]} !important;
        }}

        [data-testid="stDataFrame"] div {{
            color: {theme["text"]} !important;
        }}

        table {{
            background-color: {theme["background"]} !important;
            color: {theme["text"]} !important;
        }}

        thead tr th {{
            background-color: {theme["secondary_background"]} !important;
            color: {theme["text"]} !important;
        }}

        tbody tr td {{
            background-color: {theme["background"]} !important;
            color: {theme["text"]} !important;
        }}

        hr {{
            border-color: {theme["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    

# ---------------------------------------------------
# --- 9. App heirarchy helper
# ---------------------------------------------------

def section_card(title, subtitle=None):
    theme = THEMES[st.session_state.get("theme_mode", "light")]
    border = theme["border"]
    bg = theme["secondary_background"]
    text = theme["text"]
    
    st.markdown(
        f"""
        <div style="
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            background-color: rgba(128,128,128,0.06);
            margin-top: 1rem;
            margin-bottom: 1rem;
        ">
            <h2 style="margin-bottom: 0.6rem;">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(subtitle)


# ---------------------------------------------------
# --- 10. Latent array helpers
# ---------------------------------------------------
# Latents arrive as (n_steps, n_nodes, latent_dim) whatever the model. Nodes
# with no data (land for an ocean model) and channels that do not exist at a
# step (ragged U-Net levels) are NaN, so every statistic below works on the
# finite subset and writes NaN back everywhere else.


def format_model_time(timestamp, year_offset=0, fmt="%Y-%m-%d %H:%M"):
    """Format a timestamp in the model's own years.

    Control runs use model years such as 0425, which pandas cannot hold, so the
    extraction script stores them shifted by whole centuries. ``year_offset`` is
    that shift; it is taken off again here so the UI shows the model's dates.
    """
    if not year_offset:
        return timestamp.strftime(fmt)
    year = f"{timestamp.year - int(year_offset):04d}"
    return timestamp.strftime(fmt.replace("%Y", year))


def step_view(latent, step):
    """Split one step of a latent array into its usable nodes and channels.

    Returns:
        (node_has_data, channel_idx) where ``node_has_data`` is a boolean mask
        over nodes and ``channel_idx`` indexes the channels that exist at this
        step at every node that has data.
    """
    step_latents = np.asarray(latent[step])
    finite = np.isfinite(step_latents)
    node_has_data = finite.any(axis=1)
    if not node_has_data.any():
        return node_has_data, np.empty(0, dtype=int)
    channel_present = finite[node_has_data].all(axis=0)
    return node_has_data, np.flatnonzero(channel_present)


def nanmax_abs(values, axis):
    """Maximum absolute value along an axis, returning NaN for empty slices."""
    values = np.abs(np.asarray(values, dtype=float))
    finite_any = np.isfinite(values).any(axis=axis)
    filled = np.where(np.isfinite(values), values, -np.inf)
    out = np.max(filled, axis=axis)
    return np.where(finite_any, out, np.nan)


def scatter_to_nodes(values, node_mask, n_nodes, n_cols=None):
    """Place per-valid-node results back into a full-length NaN array."""
    if n_cols is None:
        full = np.full(n_nodes, np.nan)
    else:
        full = np.full((n_nodes, n_cols), np.nan)
    full[node_mask] = values
    return full


def cosine_similarity_to(reference, matrix):
    """Cosine similarity between one latent vector and each row of ``matrix``."""
    reference = np.asarray(reference, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    ref_norm = np.linalg.norm(reference)
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * ref_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        sims = (matrix @ reference) / denominator
    return np.where(denominator > 0, sims, np.nan)
