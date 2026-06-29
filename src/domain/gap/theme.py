"""Trail / Earthy color theme shared by the app UI and the GAP plots.

One source of truth for every color used across TrailMetrics, so the
Streamlit chrome (see ``.streamlit/config.toml``) and the Plotly figures
stay visually coherent.
"""

# --- Accent palette --------------------------------------------------------
# Brighter accents used for highlights, gradients and extra contrast. They
# stay inside the Trail / Earthy family (greens, terracotta, warm amber).
PRIMARY = "#2E6F40"          # forest green (matches the Streamlit primary)
TERRACOTTA = "#C65D3B"       # warm clay
SUNRISE = "#E8A33D"          # warm amber — the punchy highlight
MOSS = "#5E9C4E"             # mid green, bridges primary -> amber in gradients

# Left-to-right gradient used by the fun loader and plot fills.
GRADIENT = [PRIMARY, MOSS, TERRACOTTA, SUNRISE]

# --- Curve roles -----------------------------------------------------------
# Each personalized/reference curve gets a stable color so the same concept
# looks the same on every figure.
EFFICIENCY = "#2E6F40"       # forest green  -> "You (Efficiency model)"
AUTO_LEARNING = "#C65D3B"    # terracotta    -> "You (Auto-Learning model)"
BALANCED_RUNNER = "#A6843E"  # deep sand     -> reference (dashed)
KILIAN = "#3A332B"           # dark stone    -> reference (dashed)

LOW_INTENSITY = "#7FB069"    # fresh sage
HIGH_INTENSITY = "#14532B"   # deep pine

# --- Figure chrome ---------------------------------------------------------
FIGURE_FACE = "#FBF8F3"      # warm off-white (matches Streamlit background)
AXES_FACE = "#FFFDF9"        # slightly lighter than the page for contrast
GRID = "#CFC3AE"             # muted sand grid lines (a touch darker)
TEXT = "#241F19"             # deep warm near-black
SPINE = "#B8AC97"

# Distinct colors for overlaying several time scales on a single figure. Hues
# are spread out (green / clay / amber / lake / plum) so up to five scales stay
# easy to tell apart while keeping the warm Trail / Earthy feel.
TIME_SCALE_CYCLE = [
    "#2E6F40",  # forest green
    "#C65D3B",  # terracotta
    "#E8A33D",  # warm amber
    "#3A6EA5",  # lake blue
    "#7A4E9E",  # plum
]

# Cycle used for any curve that doesn't carry an explicit color.
CURVE_CYCLE = [
    EFFICIENCY,
    AUTO_LEARNING,
    SUNRISE,
    KILIAN,
    LOW_INTENSITY,
    HIGH_INTENSITY,
    BALANCED_RUNNER,
]
