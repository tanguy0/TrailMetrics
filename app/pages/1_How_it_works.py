"""How TrailMetrics works — explanations kept off the main page."""

from _helpers import add_repo_root_to_path, inject_theme_css, t

add_repo_root_to_path()

import streamlit as st

st.set_page_config(page_title=t("page.howitworks.tab"), page_icon="❓", layout="wide")
inject_theme_css()

st.title(t("page.howitworks.title"))

st.markdown(t("howitworks.body"))
