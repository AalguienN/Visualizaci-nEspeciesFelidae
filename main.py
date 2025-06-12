import streamlit as st

from funcs import show_heatmap_page, show_choropleth_page,show_charts_page, show_combined_corr_page, show_habitat_boxplot_page,show_about

st.set_page_config(layout="wide")
menu = st.sidebar.radio("🔍 Navegación", [
    "🐈🔥Heatmap avistamientos",
    "🗺Mapa de coropletas",
    "📅Gráficas temporales",
    "📊Correlación entre especies",
    "🌵Avistamientos por hábitat",
    "About",
])

if menu == "🐈🔥Heatmap avistamientos":
    show_heatmap_page()
elif menu == "🗺Mapa de coropletas":
    show_choropleth_page()
elif menu == "📅Gráficas temporales":
    show_charts_page()
elif menu == "📊Correlación entre especies":
    show_combined_corr_page()
elif menu =="About":
    show_about()
else:
    show_habitat_boxplot_page()
