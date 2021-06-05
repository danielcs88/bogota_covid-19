# %%
import json
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.io as pio

with urlopen(
    "https://raw.githubusercontent.com/danielcs88/bogota_covid-19/master/datos/bogota.json"
) as response:
    l_dads = json.load(response)


df = pd.read_csv("datos/bog_latest.csv", dtype={"codigo": str})


fig = px.choropleth_mapbox(
    df,
    geojson=l_dads,
    locations="codigo",
    featureidkey="properties.codigo",
    color="ML",
    color_continuous_scale=[
        (0, "green"),
        (0.5, "rgb(135, 226, 135)"),
        (0.5, "rgb(226, 136, 136)"),
        (1, "red"),
    ],
    hover_name="localidad",
    hover_data=[
        "fecha",
        "población",
        "casos",
        "tasa_casos_por_población",
        "densidad_hab_km2",
    ],
    range_color=(0, 2),
    mapbox_style="carto-positron",
    zoom=10.87012783741688,
    center={"lat": 4.629305577328296, "lon": -74.09870014417959},
    opacity=0.8,
    labels={
        "codigo": "Código",
        "ML": "Valor más probable de Rₜ",
        "fecha": "Fecha",
        "población": "Población",
        "casos": "Número de casos",
        "tasa_casos_por_población": "Tasa: Casos por Población",
        "densidad_hab_km2": "Densidad de habitante por km²",
    },
    width=1024,
    height=563,
)

fig.layout.font.family = "Arial"

fig.update_layout(
    title="Bogotá | Mapa Rₜ por Localidad",
    width=1000,
    height=1000,
    annotations=[
        dict(
            xanchor="right",
            x=1,
            yanchor="top",
            y=-0.05,
            showarrow=False,
            text="Fuentes: Secretaría Distrital de Salud: Datos Abiertos Bogotá, DANE",
        )
    ],
)
fig.show()


# %%
# pio.write_json(fig, "fl_choro.json")


# %%
with open("../danielcs88.github.io/html/rt_bogota.html", "w") as f:
    f.write(fig.to_html(include_plotlyjs="cdn"))


# %%
fig2 = px.choropleth_mapbox(
    df,
    geojson=l_dads,
    locations="codigo",
    featureidkey="properties.codigo",
    color="casos",
    color_continuous_scale="Reds",
    hover_name="localidad",
    hover_data=[
        "fecha",
        "población",
        "casos",
        "tasa_casos_por_población",
        "densidad_hab_km2",
        "ML",
    ],
    mapbox_style="carto-positron",
    zoom=10.87012783741688,
    center={"lat": 4.629305577328296, "lon": -74.09870014417959},
    opacity=0.8,
    labels={
        "codigo": "Código",
        "ML": "Valor más probable de Rₜ",
        "fecha": "Fecha",
        "población": "Población",
        "casos": "Número de casos",
        "tasa_casos_por_población": "Tasa: Casos por Población",
        "densidad_hab_km2": "Densidad de habitante por km²",
    },
    width=1024,
    height=563,
)

fig2.layout.font.family = "Arial"

fig2.update_layout(
    width=1000,
    height=1000,
    title="Bogotá | Casos por Localidad",
    annotations=[
        dict(
            xanchor="right",
            x=1,
            yanchor="top",
            y=-0.05,
            showarrow=False,
            text="Fuentes: Secretaría Distrital de Salud: Datos Abiertos Bogotá, DANE",
        )
    ],
)
fig2.show()


# %%
with open("../danielcs88.github.io/html/casos_bogota.html", "w") as f:
    f.write(fig2.to_html(include_plotlyjs="cdn"))


# %%
get_ipython().system(" cd ../danielcs88.github.io/ && git pull")


# %%
get_ipython().system(
    ' cd ../danielcs88.github.io/ && git add --all && git commit -m "Update" && git push'
)


# %%
# ! git add --all && git commit -m "Update" && git push
