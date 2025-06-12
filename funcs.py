import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
import json
import altair as alt
from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster



# Para GeoJSON mundial de países:
# Descárgalo de alguno de estos repositorios:
# - https://github.com/datasets/geo-countries/blob/master/data/countries.geojson
# - https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json
# Guarda el archivo como 'world_countries.geojson' en tu directorio de trabajo

# ---------- Data loading functions ----------

@st.cache_data
def load_occurrence():
    """Carga gbifID, scientificName y vernacularName"""
    return pd.read_csv(
        r".\gatos_light.csv",
        sep="\t",
        usecols=["gbifID", "scientificName", "vernacularName"],
        dtype=str
    )

@st.cache_data
def load_multimedia():
    """Carga gbifID e identifier para multimedia"""
    return pd.read_csv(
        r".\multimedia.txt",
        sep="\t",
        usecols=["gbifID", "identifier"],
        dtype=str
    )

@st.cache_data
def load_df():
    return pd.read_csv(
        r".\gatos_light.csv",
        sep="\t",
        usecols=[
            "countryCode","scientificName","vernacularName",
            "decimalLatitude","decimalLongitude",
            "eventDate","year","month","habitat",
            "elevation","depth",
            "coordinateUncertaintyInMeters","individualCount"
        ],
        dtype=str
    )

@st.cache_data
def get_species(df, min_occ):
    """Devuelve lista de nombres comunes con al menos min_occ registros georreferenciados"""
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"]).copy()
    df["scientificName"] = df["scientificName"].fillna(df["scientificName"])
    counts = df["scientificName"].value_counts()
    return sorted(counts[counts >= min_occ].index)

@st.cache_data
def filter_dataframe(
    df: pd.DataFrame,
    filter_col: str,
    filter_val,
    numeric_cols: list[str] = None
) -> pd.DataFrame:
    """
    Filtra df donde df[filter_col] == filter_val, elimina nulos en
    numeric_cols y convierte esas columnas a float.

    :param df: DataFrame de entrada
    :param filter_col: nombre de la columna por la que filtrar
    :param filter_val: valor que debe tener filter_col
    :param numeric_cols: lista de columnas a convertir a float
    :return: DataFrame filtrado y convertido
    """
    df = df.copy()
    # Rellenar nulos en la columna de filtro si es necesario
    df[filter_col] = df[filter_col].fillna(df[filter_col])
    
    # Aplicar máscara
    mask = df[filter_col] == filter_val
    f = df.loc[mask].copy()
    
    # Si nos piden columnas numéricas, limpiamos nulos y casteamos
    if numeric_cols:
        for col in numeric_cols:
            if col in f:
                f = f[f[col].notna()]
                f[col] = f[col].astype(float)
    
    return f


@st.cache_data
def get_img_url(mult, vernac):
    """Obtiene la primera URL de imagen asociada al nombre común"""
    occ = load_occurrence()
    merged = mult.merge(
        occ[["gbifID", "scientificName", "vernacularName"]],
        on="gbifID", how="left"
    )
    merged["scientificName"] = merged["scientificName"].fillna(merged["scientificName"])
    fila = merged[merged["scientificName"] == vernac]
    if not fila.empty and pd.notna(fila["identifier"].iloc[0]):
        return fila["identifier"].iloc[0]
    return None

def show_image(seleccion):
    mult = load_multimedia()
    img_url = get_img_url(mult, seleccion)
    if img_url:
        st.image(img_url, use_container_width=True, caption=seleccion)
    else:
        st.info("No hay imagen disponible para esta especie.")

def show_heatmap_page():
    """Página de Streamlit: Mapa de calor de avistamientos"""
    st.title("🐈 Avistamientos - Mapa de calor")
    
    coords_df = load_df()

    # Control del umbral de observaciones
    min_occ = st.sidebar.number_input(
        "Mínimo nº de avistamientos:", 1, 2000, 1000, step=1
    )
    especies = get_species(coords_df, min_occ)
    seleccion = st.selectbox("Selecciona una especie (nombre común):", especies)

    f_coords = filter_dataframe(
        coords_df,
        filter_col="scientificName",
        filter_val=seleccion,
        numeric_cols=["decimalLatitude", "decimalLongitude"]
    )

    if not f_coords.empty:
        coors = list(zip(f_coords["decimalLatitude"], f_coords["decimalLongitude"]))
        centro = [f_coords["decimalLatitude"].mean(), f_coords["decimalLongitude"].mean()]
        mapa = folium.Map(location=centro, zoom_start=2, tiles="CartoDB dark_matter")
        HeatMap(coors, radius=5, blur=5, max_zoom=12).add_to(mapa)

        col1, col2 = st.columns([3, 1], gap="small")
        with col1:
            st.markdown(f"**Especie mostrada:** {seleccion} ({len(coors)} registros)")
            st_folium(mapa, width=900, height=600, returned_objects=[], key=f"heatmap_{seleccion}")
        with col2:
            show_image(seleccion)
            st.subheader("Frecuencia de hábitats")

            # 1) Cuenta y prepara el DataFrame con columnas claras
            hab_counts = (
                f_coords["habitat"]
                .dropna()
                .value_counts()
                .head(10)
                .rename_axis("habitat")           # convierte el índice en columna "habitat"
                .reset_index(name="conteo")       # la serie de valores pasa a columna "conteo"
            )

            # Ahora hab_counts tiene dos columnas:
            #   - "habitat" (string)
            #   - "conteo"   (numérico)
            # 2) Gráfico de barras horizontales
            bar_hab = (
                alt.Chart(hab_counts)
                .mark_bar()
                .encode(
                    y=alt.Y("habitat:O", sort=alt.EncodingSortField(field="conteo", order="descending"), title="Hábitat"),
                    x=alt.X("conteo:Q", title="Número de avistamientos"),
                    tooltip=["habitat","conteo"]
                )
                .properties(height=300, title="Top 10 hábitats por avistamientos")
            )
            st.altair_chart(bar_hab, use_container_width=True)

    else:
        st.warning("No se encontraron registros georreferenciados para esta especie.")

import pycountry

def alpha2_to_alpha3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except:
        return None

def show_choropleth_page():
    """Página de Streamlit: Mapa coroplético mundial por especie y país"""
    st.title("🐈 Avistamientos - Mapa coroplético por especie y país")
    
    coords_df = load_df()
    coords_df["scientificName"] = coords_df["scientificName"].fillna(coords_df["scientificName"])
    
    min_occ_species = st.sidebar.number_input(
        "Mínimo nº de avistamientos por especie:", 1, 2000, 1000, step=1
    )

    especies = get_species(coords_df, min_occ_species)
    seleccion = st.selectbox("Selecciona una especie (nombre común):", especies)

    df = filter_dataframe(
        coords_df,
        filter_col="scientificName",
        filter_val=seleccion,
        numeric_cols=["decimalLatitude", "decimalLongitude"]
    )
    

    if not df.empty:
        df["countryCode"] = df["countryCode"].str.upper().str.strip()
        df["countryCode3"] = df["countryCode"].apply(lambda x: alpha2_to_alpha3(x) if pd.notna(x) else None)
        
        agg = (
            df
            .dropna(subset=["countryCode3"])
            .groupby("countryCode3")
            .size()
            .reset_index(name="count")
        )

        if agg.empty:
            st.warning(f"No hay registros georreferenciados para {seleccion}.")
            return
        
        geo_path = r"./countries2.geojson"
        
        world_map = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
        folium.Choropleth(
            geo_data=geo_path,
            data=agg,
            columns=["countryCode3", "count"],
            key_on="feature.id",
            fill_color="YlGnBu",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"Avistamientos de {seleccion}"
        ).add_to(world_map)
        

        col_map, col_chart = st.columns([3,1], gap="small")
        with col_map:
            st.markdown(f"### Especie: **{seleccion}** — {agg['count'].sum()} registros totales")
            st_folium(world_map,  width=900, height=600, returned_objects=[], key="choropleth_world")
        with col_chart:
            st.subheader("Avistamientos por país")
            agg_sorted = agg.sort_values("count", ascending=False)
            agg_sorted["countryName"] = agg_sorted["countryCode3"].apply(
                lambda x: pycountry.countries.get(alpha_3=x).name if pycountry.countries.get(alpha_3=x) else x
            )
            agg_sorted = agg_sorted.set_index("countryName")

            df_chart = agg_sorted.reset_index().rename(columns={"index":"countryName"})

            chart = (
                alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X(
                        "countryName:N",
                        sort=alt.EncodingSortField(field="count", order="descending"),
                        title="País"
                    ),
                    y=alt.Y("count:Q", title="Avistamientos")
                ).properties(width=400, height=500)
            )

            show_image(seleccion)
            st.altair_chart(chart, use_container_width=True)

def show_charts_page():
    st.title("🐈 Gráficas – Avistamientos por época del año")
    
    # 1) Umbral y carga unificada de especies
    min_occ = st.sidebar.number_input("Mínimo nº de avistamientos:", 1, 2000, 1000)
    df_all = load_df()
    especies = get_species(df_all, min_occ)
    seleccion = st.selectbox("Selecciona una especie (nombre común):", especies)
    
    # 2) Filtrado por especie y limpieza de month/year
    df_sel = df_all[df_all["scientificName"].fillna(df_all["scientificName"]) == seleccion]
    df_sel = df_sel.dropna(subset=["month", "year"])
    df_sel["month"] = df_sel["month"].astype(int)
    df_sel["year"]  = df_sel["year"].astype(int)
    if df_sel.empty:
        st.warning("No hay registros para esa especie.")
        return


    col_map, col_2, col_3 = st.columns([1,1,1], gap="small")

        
    
    with col_2:
        # 3) Avistamientos por mes (gráfico de línea)
        monthly = (
            df_sel.groupby("month")
                .size()
                .reset_index(name="avistamientos")
                .sort_values("month")
        )
        meses = ["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"]
        monthly["mes"] = monthly["month"].map(lambda m: meses[m-1])
        line_month = (
            alt.Chart(monthly)
            .mark_bar()
            .encode(x=alt.X("mes:N", sort=meses, title="Mes"),
                    y=alt.Y("avistamientos:Q", title="Avistamientos"))
            .properties(height=300)
        )
        st.altair_chart(line_month, use_container_width=True)

        df_sel["month_num"] = df_sel["month"].astype(int)
        meses_disponibles = sorted(df_sel["month"].unique())
        min_m, max_m = meses_disponibles[0], meses_disponibles[-1]


        # 4) Tendencia anual (línea)
        yearly = (
            df_sel.groupby("year")
                .size()
                .reset_index(name="avistamientos")
                .sort_values("year")
        )
        line_year = (
            alt.Chart(yearly)
            .mark_line(point=True)
            .encode(x=alt.X("year:O", title="Año"),
                    y=alt.Y("avistamientos:Q", title="Avistamientos"))
            .properties(height=300)
        )
        st.altair_chart(line_year, use_container_width=True)

    with col_3:
        show_image(seleccion)

        # 5) Heatmap mes vs año
        pivot = (
            df_sel.groupby(["year","month"])
                .size()
                .reset_index(name="avistamientos")
        )
        pivot["mes"] = pivot["month"].map(lambda m: meses[m-1])
        heatmap = (
            alt.Chart(pivot)
            .mark_rect()
            .encode(
                x=alt.X("mes:N", sort=meses, title="Mes"),
                y=alt.Y("year:O", title="Año"),
                color=alt.Color("avistamientos:Q", title="Avistamientos"),
                tooltip=["year","mes","avistamientos"]
            )
            .properties(height=400)
        )
        st.altair_chart(heatmap, use_container_width=True)




    with col_map:

        start_month, end_month = st.slider(
            "Rango de meses (sólo donde hay datos):",
            min_value=min_m, max_value=max_m,
            value=(min_m, max_m)
        )

        df_f = df_sel[
            (df_sel["month_num"] >= start_month) &
            (df_sel["month_num"] <= end_month)
        ]

        if df_f.empty:
            st.warning(
                f"No hay avistamientos de **{seleccion}** "
                f"entre los meses {start_month} y {end_month}."
            )
            return

        coors  = list(zip(
            df_f["decimalLatitude"].astype(float),
            df_f["decimalLongitude"].astype(float)
        ))
        centro = [
            df_f["decimalLatitude"].astype(float).mean(),
            df_f["decimalLongitude"].astype(float).mean()
        ]

        m = folium.Map(location=centro, zoom_start=2, tiles="CartoDB dark_matter")
        cluster = MarkerCluster(
            singleMarkerMode=True,   # usa el icono de clúster (círculo) siempre, incluso para 1 punto
            # opcional: si quisieras personalizar el HTML o el tamaño, podrías añadir aquí
            # icon_create_function=... 
        )

        # añade los marcadores individuales al clúster
        for lat, lon in coors:
            folium.Marker(location=[lat, lon]).add_to(cluster)

        # añade el clúster al mapa
        cluster.add_to(m)

        st.markdown(f"**Especie:** {seleccion} — “{start_month}–{end_month}” ({len(coors)} puntos)")
        st_folium(
            m, width=400, height=400,
            returned_objects=[],   # evita reruns por pane/zoom
            center=centro, zoom=2
        )

    
def show_charts_2_page():
    st.title("🐈 Gráficas – Avistamientos por época del año")
    
    # 1) Umbral y carga unificada de especies
    min_occ = st.sidebar.number_input("Mínimo nº de avistamientos:", 1, 2000, 1000)
    df_all = load_df()
    especies = get_species(df_all, min_occ)
    seleccion = st.selectbox("Selecciona una especie (nombre común):", especies)
    
    # 1) Pivot habitat × especie
    hab_pivot = (
        df_all[df_all["scientificName"].isin(especies) & df_all["habitat"].notna()]
        .groupby(["habitat","scientificName"])
        .size()
        .reset_index(name="cnt")
        .pivot(index="habitat", columns="scientificName", values="cnt")
        .fillna(0)
    )

    # 1) Calcula la matriz de correlación
    corr_h = hab_pivot.corr()

    # 2) Renombra los ejes (index y columns) para que no colisionen
    corr_h.index.name   = "especie1"
    corr_h.columns.name = "especie2"

    # 3) Ahora el stack + reset_index funciona sin error
    corr_h_long = corr_h.stack().reset_index(name="ρ")


    # 3) Heatmap
    heat_h = (
        alt.Chart(corr_h_long)
           .mark_rect()
           .encode(
               x=alt.X("especie1:N", sort=especies, title="Especie 1"),
               y=alt.Y("especie2:N", sort=especies, title="Especie 2"),
               color=alt.Color("ρ:Q", scale=alt.Scale(scheme="magma"), title="ρ"),
               tooltip=["especie1","especie2","ρ"]
           )
           .properties(
               title="Corr. de uso de hábitats",
               width=1000,    # ancho fijo
               height=1000    # alto fijo, igual al ancho
           )
    )
    # Quita use_container_width para que respete width/height
    st.altair_chart(heat_h, use_container_width=False)

    #-------------------------------------

    n_lat_bins = st.sidebar.slider("Número de bins de latitud:", 5, 50, 20)
    n_lon_bins = st.sidebar.slider("Número de bins de longitud:", 5, 50, 20)

    df = df_all.copy()
    # Asegúrate de convertir a float y eliminar nulos
    df = df.dropna(subset=["decimalLatitude","decimalLongitude"])
    df["decimalLatitude"]  = df["decimalLatitude"].astype(float)
    df["decimalLongitude"] = df["decimalLongitude"].astype(float)

    # 1) Crear bins de latitud y longitud
    df["lat_bin"] = pd.cut(df["decimalLatitude"], bins=n_lat_bins)
    df["lon_bin"] = pd.cut(df["decimalLongitude"], bins=n_lon_bins)

    # 2) Elegir si quieres analizar latitud o longitud
    axis = st.sidebar.radio("¿Correlacionar sobre…", ["latitud", "longitud"])
    bin_col = "lat_bin" if axis=="latitud" else "lon_bin"

    # 3) Pivot bin × especie
    pivot = (
        df[df["scientificName"].isin(especies)]
        .groupby([bin_col, "scientificName"])
        .size()
        .reset_index(name="cnt")
        .pivot(index=bin_col, columns="scientificName", values="cnt")
        .fillna(0)
    )

    # 4) Matriz de correlación
    corr = pivot.corr()
    corr.index.name   = "especie1"
    corr.columns.name = "especie2"

    # 5) Enmascara mitad superior (opcional)
    import numpy as np
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    corr = corr.where(mask)

    # 6) Aplana para Altair
    corr_long = corr.stack(dropna=True).reset_index(name="ρ")

    # 7) Dibuja heatmap
    heat = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("especie1:N", sort=especies, title="Especie 1"),
            y=alt.Y("especie2:N", sort=especies, title="Especie 2"),
            color=alt.Color("ρ:Q", scale=alt.Scale(scheme="magma"), title="ρ"),
            tooltip=["especie1","especie2","ρ"]
        )
        .properties(
            title=f"Corr. espacial sobre {axis} (bins={n_lat_bins if axis=='latitud' else n_lon_bins})",
            width=800, height=800
        )
    )
    st.altair_chart(heat, use_container_width=False)




import numpy as np

def show_combined_corr_page():
    st.title("🐈 Comparativa de correlaciones entre especies")

    # 1) Parámetros y carga
    min_occ = st.sidebar.number_input("Mínimo nº de avistamientos:", 1, 2000, 1000)
    df_all = load_df()
    especies = get_species(df_all, min_occ)

    # 2) Función helper para generar corr_long de cualquier pivot
    def compute_corr_long(pivot_df):
        corr = pivot_df.corr()
        corr.index.name = "especie1"
        corr.columns.name = "especie2"
        # Quedamos con mitad superior para no duplicar
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        corr = corr.where(mask)
        return corr.stack(dropna=True).reset_index(name="ρ")

    # ── a) Hábitat
    hab_pivot = (
        df_all[df_all["scientificName"].isin(especies) & df_all["habitat"].notna()]
        .groupby(["habitat","scientificName"]).size()
        .unstack(fill_value=0)
    )
    corr_h = compute_corr_long(hab_pivot)
    corr_h["tipo"] = "hábitat"

    # ── b) Latitud binned
    df = df_all.dropna(subset=["decimalLatitude","decimalLongitude"]).copy()
    df["decimalLatitude"] = df["decimalLatitude"].astype(float)
    df["lat_bin"] = pd.cut(df["decimalLatitude"], bins=20).astype(str)
    lat_pivot = (
        df[df["scientificName"].isin(especies)]
        .groupby(["lat_bin","scientificName"]).size()
        .unstack(fill_value=0)
    )
    corr_lat = compute_corr_long(lat_pivot)
    corr_lat["tipo"] = "latitud"

    # ── c) Longitud binned
    df["decimalLongitude"] = df["decimalLongitude"].astype(float)
    df["lon_bin"] = pd.cut(df["decimalLongitude"], bins=20).astype(str)
    lon_pivot = (
        df[df["scientificName"].isin(especies)]
        .groupby(["lon_bin","scientificName"]).size()
        .unstack(fill_value=0)
    )
    corr_lon = compute_corr_long(lon_pivot)
    corr_lon["tipo"] = "longitud"

    # 3) Unir los tres DataFrames
    corr_all = pd.concat([corr_h, corr_lat, corr_lon], ignore_index=True)


    # 3.5) Top relaciones por correlación
    top_n = st.sidebar.number_input("Número de relaciones a mostrar:", 5, 50, 10)
    # Ordenamos de mayor a menor ρ
    top_rel = (
        corr_all
        .sort_values("ρ", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    st.subheader(f"Top {top_n} correlaciones más altas")
    st.table(top_rel[["tipo","especie1","especie2","ρ"]])


    # 4) Dibujo facetado
    heat = (
        alt.Chart(corr_all)
           .mark_rect()
           .encode(
               x=alt.X("especie1:N", sort=especies, title="Especie 1"),
               y=alt.Y("especie2:N", sort=especies, title="Especie 2"),
               color=alt.Color("ρ:Q", scale=alt.Scale(scheme="magma"), title="ρ"),
               tooltip=["tipo","especie1","especie2","ρ"]
           )
           .properties(width=300, height=300)
           .facet(
               column=alt.Column("tipo:N", title=None, header=alt.Header(labelAngle=0))
           )
    )
    st.altair_chart(heat, use_container_width=True)



    import networkx as nx
    import plotly.graph_objects as go

    # 1) Elige un umbral mínimo de ρ
    threshold = st.sidebar.slider(
        "Umbral mínimo de correlación (ρ):", 0.0, 1.0, 0.5, step=0.05
    )

    # 2) Filtra las relaciones por encima del umbral
    edges = corr_all[corr_all["ρ"] >= threshold]

    # 3) Construye el grafo
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["especie1"], row["especie2"], weight=row["ρ"])

    # 4) Calcula posiciones con spring layout
    pos = nx.spring_layout(G, seed=42)

    # 5) Prepara las trazas de aristas
    edge_x, edge_y = [], []
    edge_width = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        # Ancho de línea proporcional a la correlación
        edge_width.append(d["weight"] * 5)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none"
    )

    # 6) Prepara la traza de nodos
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=12,
            color="steelblue",
            line=dict(width=2, color="white")
        )
    )

    # 7) Monta la figura
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Red de correlaciones (ρ ≥ {threshold})",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    # 8) Muestra en Streamlit
    st.plotly_chart(fig, use_container_width=True)



def show_habitat_boxplot_page():
    st.title("🐈 Boxplot de avistamientos por hábitat")
    
    # 1) Parámetros
    min_occ = st.sidebar.number_input(
        "Mínimo nº de avistamientos por especie:", 
        1, 2000, 1000, step=1
    )
    top_n = st.sidebar.number_input(
        "Número de hábitats a mostrar (con más especies):", 
        1, 100, 10, step=1
    )
    
    # 2) Carga y filtrado de especies con al menos min_occ registros
    df_all = load_df()
    especies = get_species(df_all, min_occ)
    df = df_all[
        df_all["scientificName"].fillna(df_all["scientificName"]).isin(especies)
    ].dropna(subset=["habitat"])

    # df = df_all[df_all["habitat"] == "Savanna"]

    # Excluir valores desconocidos
    df = df[df["habitat"].notna() & ~df["habitat"].isin(["unknown","[]","NO DISPONIBLE"])]

    # 3) Selección de los hábitats con mayor número de especies distintas
    top_habitats = (
        df.groupby("habitat")["scientificName"]
          .nunique()
          .sort_values(ascending=False)
          .head(top_n)
          .index
          .tolist()
    )
    df = df[df["habitat"].isin(top_habitats)]
    
    # 4) Cómputo de conteos por (hábitat, especie)
    hab_counts = (
        df
        .groupby(["habitat","scientificName"])
        .size()
        .reset_index(name="count")
    )
    
    # 4.1) Eliminación de outliers por hábitat (átomo de IQR)
    hab_counts = (
        hab_counts
        .groupby("habitat", group_keys=False)
        .apply(lambda g: g[
            (g["count"] >= g["count"].quantile(0.25) - 1.5 * (g["count"].quantile(0.75) - g["count"].quantile(0.25)))
            & (g["count"] <= g["count"].quantile(0.75) + 1.5 * (g["count"].quantile(0.75) - g["count"].quantile(0.25)))
        ])
    )
    
    # 5) Boxplot horizontal con Altair (sin outliers extremos)
    box = (
        alt.Chart(hab_counts)
        .mark_boxplot()
        .encode(
            y=alt.Y(
                "habitat:N", sort=top_habitats,
                title="Hábitat",
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)
            ),
            x=alt.X(
                "count:Q",
                title="Registros por especie",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)
            ),
            tooltip=["habitat", "scientificName", "count"]
        )
        .properties(
            width=700,
            height=700,
            title="Distribución de avistamientos por especie en cada hábitat (sin outliers)"
        )
        .configure_axisLeft(
            labelFontSize=14,
            titleFontSize=16,
            labelLimit=300  # aumenta el espacio para etiquetas largas
        )
    )
    st.altair_chart(box, use_container_width=True)


def show_about():
    """
    Página 'About' con información de la aplicación,
    fuentes de datos, autor y enlaces.
    """
    st.title("ℹ️ Acerca de esta aplicación")
    st.markdown(
        "Esta aplicación interactiva muestra avistamientos de especies de gatos "
        "a nivel global. Permite explorar mapas de calor, coropletas, gráficos "
        "temporales y boxplots de distribución por hábitat."
    )
    
    st.header("Fuentes de datos y citación GBIF")
    st.markdown(
        "Los datos de avistamientos se obtuvieron de GBIF (Global Biodiversity Information Facility). "
        "GBIF.org (10 June 2025) GBIF Occurrence Download  https://doi.org/10.15468/dl.z9ryb4"
    )
    st.header("Herramientas y tecnologías")
    st.markdown(
        "- Streamlit para la interfaz web.  "
        "- Folium y `streamlit_folium` para mapas interactivos.  "
        "- Altair para visualizaciones.  "
        "- Pandas para manipulación de datos."
    )

    st.header("Autor y repositorio")
    st.markdown(
        "- **Autor:** Adrián Camacho García"
        "- **Repositorio:** [GitHub](https://github.com/tu_usuario/tu_repositorio)"
    )

    st.caption("Última actualización: 2025")
