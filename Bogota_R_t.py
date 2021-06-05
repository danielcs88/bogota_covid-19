# -*- coding: utf-8 -*-
# %%
from IPython import get_ipython

# %% [markdown]
"""
# Análisis de 🇨🇴 Bogotá a nivel de localidad | $R_t$

por Daniel Cárdenas

Si $R_t > 1$, el número de casos aumentará, como al comienzo de una epidemia.
Cuando $R_t = 1$, la enfermedad es endémica, y cuando $R_t <1$ habrá una
disminución en el número de casos.

Entonces, los epidemiólogos usan $R_t$ para hacer recomendaciones de políticas.
Es por eso que este número es tan importante.

Mi modelo es una adaptación del model de [Kevin
Systrom](https://github.com/k-sys/covid-19)

## Fuente de Datos

Mi fuente de datos es del Ministerio de Salud de Colombia y su plataforma [Casos
positivos de COVID-19 en
Colombia](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data).
"""

# %%
# from IPython.display import clear_output
# clear_output()

# %%
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num, num2date
from matplotlib.patches import Patch
from scipy import stats as sps
from scipy.interpolate import interp1d

get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic("matplotlib", "inline")


# %%
from cycler import cycler

custom = cycler(
    "color",
    [
        "#B3220F",
        "#F16E53",
        "#FFC475",
        "#006F98",
        "#1ABBEF",
        "#7FD2FD",
        "#153D53",
        "#0F9197",
    ],
)


plt.rc("axes", prop_cycle=custom)
plt.rcParams["figure.dpi"] = 140


# %%
def highest_density_interval(pmf, p=0.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if isinstance(pmf, pd.DataFrame):
        return pd.DataFrame(
            [highest_density_interval(pmf[col], p=p) for col in pmf], index=pmf.columns
        )

    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i + 1 :]):
            if (high_value - value > p) and (not best or j < best[1] - best[0]):
                best = (i, i + j + 1)
                break

    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=[f"Low_{p*100:.0f}", f"High_{p*100:.0f}"])


# %%
cronologia = "datos/cronologia.csv"

bogota = pd.read_csv(cronologia)

# Print Counties
latest_date = bogota[-1:]
latest_date = latest_date.fecha
latest_date = " ".join(str(elem) for elem in latest_date)
print(latest_date)


# %%
distritos = sorted(set(bogota.localidad.unique()))
len(distritos)


# %%
bogota.tail(len(distritos))


#  ### Filtros
#
#  * Distrito seleccionado
#  * Eliminar localidades listados como "Sin Dato" o "Fuera de Bogotá"
#  * Eliminar filas con menos de 10 casos `filtro_localidad_fila`
#  * Eliminar localidades con menos filas que `filtro_localidad` después de suavizar

# %%
# def ortografia(error):
#     correcion = error.replace(" Kennedy", "Kennedy")
#     return correcion

# %%
# bogota["localidad"] = bogota.apply(lambda x: ortografia(x["localidad"]), axis=1)

# %%
filtro_localidad = 10
filtro_localidad_fila = 10

# %%
bogota = bogota[bogota.casos >= filtro_localidad_fila].copy()
bogota = bogota[bogota.localidad != "Fuera de Bogotá"].copy()
bogota = bogota[bogota.localidad != "Sin Dato"].copy()
bogota.shape

# %%
bogota.tail()
print(len(bogota))

# %%
bogota = bogota[["fecha", "localidad", "casos"]].copy()
bogota["fecha"] = pd.to_datetime(bogota["fecha"])
bogota = bogota.set_index(["localidad", "fecha"]).squeeze().sort_index()

# %%
bogota

# %%
bogota_g = (
    bogota.groupby(["localidad"])
    .count()
    .reset_index()
    .rename({"casos": "filas"}, axis=1)
)
bogota_g


# %%
lista_localidad = bogota_g[bogota_g.filas >= filtro_localidad_fila][
    "localidad"
].tolist()
print(len(lista_localidad))

l_dad = lista_localidad.index("Fontibón")


# %%
def prepare_cases(casos, cutoff=1):
    new_cases = casos.diff()

    smoothed = (
        new_cases.rolling(7, win_type="gaussian", min_periods=1, center=True)
        .mean(std=3)
        .round()
    )

    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


def casos_nuevos(l_dad):

    casos = bogota.xs(l_dad).rename(f"Casos en {l_dad}")

    original, smoothed = prepare_cases(casos)

    original.plot(
        title=f"{l_dad} | Casos nuevos: {latest_date}",
        c="k",
        linestyle=":",
        alpha=0.5,
        label="Actual",
        legend=True,
        figsize=(500 / 72, 300 / 72),
    )

    ax = smoothed.plot(label="Promedio semanal (7 días) ", legend=True)

    ax.get_figure().set_facecolor("w")
    plt.savefig(f"gráficos/{l_dad}.svg")
    plt.clf()


# %%
casos_nuevos("Kennedy")


# %%
casos_nuevos("Bosa")


# %%
casos_nuevos("Rafael Uribe Uribe")


# %%
casos_nuevos("Engativá")


# %%
casos_nuevos("Antonio Nariño")


# %%
casos_nuevos("Barrios Unidos")


# %%
casos_nuevos("Tunjuelito")


# %%
casos_nuevos("Los Mártires")


# %%
casos_nuevos("Puente Aranda")


# %%
casos_nuevos("Suba")

# %%
casos_nuevos("Fontibón")

# %%
casos_nuevos("Teusaquillo")


# %%
casos_nuevos("San Cristóbal")


# %%
casos_nuevos("Usaquén")


# %%
casos_nuevos("Ciudad Bolívar")


# %%
casos_nuevos("Chapinero")


# %%
casos_nuevos("Santa Fe")


# %%
casos_nuevos("Usme")


# %%
l_dad = "Kennedy"


# %%
casos = bogota.xs(l_dad).rename(f"Casos en {l_dad}")

original, smoothed = prepare_cases(casos)


# %%
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316

GAMMA = 1 / 7

# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)


def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam), index=r_t_range, columns=sr.index[1:]
    )

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range, scale=sigma).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range, columns=sr.index, data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator + 1)

    return posteriors, log_likelihood


# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=0.25)


# %%
ax = posteriors.plot(
    title=f"{l_dad} \n Posteriores diarios de $R_t$ \n {latest_date}",
    legend=False,
    lw=1,
    c="k",
    alpha=0.3,
    xlim=(0.4, 6),
)

ax.set_xlabel("$R_t$")


# %%
# Note that this takes a while to execute - it's not the most efficient algorithm
hdis = highest_density_interval(posteriors, p=0.9)

most_likely = posteriors.idxmax().rename("ML")

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

result.tail()


# %%
def plot_rt(result, ax, county_name):

    ax.set_title(f"{l_dad}")

    # Colors
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(
        np.r_[np.linspace(BELOW, MIDDLE, 25), np.linspace(MIDDLE, ABOVE, 25)]
    )
    color_mapped = lambda y: np.clip(y, 0.5, 1.5) - 0.5

    index = result["ML"].index.get_level_values("fecha")
    values = result["ML"].values

    # Plot dots and line
    ax.plot(index, values, c="k", zorder=1, alpha=0.25)
    ax.scatter(
        index,
        values,
        s=40,
        lw=0.5,
        c=cmap(color_mapped(values)),
        edgecolors="k",
        zorder=2,
    )

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(
        date2num(index),
        result["Low_90"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )

    highfn = interp1d(
        date2num(index),
        result["High_90"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )

    extended = pd.date_range(
        start=pd.Timestamp("2020-03-01"), end=index[-1] + pd.Timedelta(days=1)
    )

    ax.fill_between(
        extended,
        lowfn(date2num(extended)),
        highfn(date2num(extended)),
        color="k",
        alpha=0.1,
        lw=0,
        zorder=3,
    )

    ax.axhline(1.0, c="k", lw=1, label="$R_t=1.0$", alpha=0.25)

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(0)
    ax.grid(which="major", axis="y", c="k", alpha=0.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(
        pd.Timestamp("2020-03-01"),
        result.index.get_level_values("fecha")[-1] + pd.Timedelta(days=1),
    )
    fig.set_facecolor("w")


fig, ax = plt.subplots(figsize=(600 / 72, 400 / 72))

plot_rt(result, ax, l_dad)
ax.set_title(f"{l_dad} | $R_t$ \n {latest_date}")
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# plt.savefig(f"gráficos/{l_dad}_rt.svg")


# %%
sigmas = np.linspace(1 / 20, 1, 20)

targets = bogota.index.get_level_values("localidad").isin(lista_localidad)
l_dad_proceso = bogota.loc[targets]

results = {}
failed_bogota = []
skipped_bogota = []

for l_dad, casos in l_dad_proceso.groupby(level="localidad"):

    print(l_dad)
    new, smoothed = prepare_cases(casos, cutoff=1)

    if len(smoothed) < 5:
        skipped_bogota.append(l_dad)
        continue

    result = {}

    # Holds all posteriors with every given value of sigma
    result["posteriors"] = []

    # Holds the log likelihood across all k for each value of sigma
    result["log_likelihoods"] = []

    try:
        for sigma in sigmas:
            posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
            result["posteriors"].append(posteriors)
            result["log_likelihoods"].append(log_likelihood)
        # Store all results keyed off of state name
        results[l_dad] = result
    #         clear_output(wait=True)
    except:
        failed_bogota.append(l_dad)
        print(f"Posteriors failed for {l_dad}")

print(f"Posteriors failed for {len(failed_bogota)} localidads: {failed_bogota}")
print(f"Skipped {len(skipped_bogota)} localidads: {skipped_bogota}")
print(f"Continuing with {len(results)} counties / {len(lista_localidad)}")
print("Done.")


# %%
# Each index of this array holds the total of the log likelihoods for
# the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for dpto, result in results.items():
    total_log_likelihoods += result["log_likelihoods"]

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()
# print(max_likelihood_index)

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]

# Plot it
fig, ax = plt.subplots()
ax.set_title(f"Valor de probabilidad máxima para $\sigma$ = {sigma:.2f}")
ax.plot(sigmas, total_log_likelihoods)
ax.axvline(sigma, color="k", linestyle=":")

# %% [markdown]
"""
### Compilar resultados finales

Dado que hemos seleccionado el óptimo $\sigma$, tomemos la parte posterior
precalculada correspondiente a ese valor de $\sigma$ para cada departamento.
Calculemos también los intervalos de densidad más alta del 90% y 50% (esto lleva
un poco de tiempo) y también el valor más probable.
"""

# %%
final_results = None
hdi_error_list = []

for l_dad, result in results.items():
    print(l_dad)
    try:
        posteriors = result["posteriors"][max_likelihood_index]
        hdis_90 = highest_density_interval(posteriors, p=0.9)
        hdis_50 = highest_density_interval(posteriors, p=0.5)
        most_likely = posteriors.idxmax().rename("ML")
        result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
        if final_results is None:
            final_results = result
        else:
            final_results = pd.concat([final_results, result])
        clear_output(wait=True)
    except:
        print(f"HDI failed for {l_dad}")
        hdi_error_list.append(l_dad)
        pass

print(f"HDI error list: {hdi_error_list}")
print("Done.")

# %% [markdown]
"""
### Trazar todos los departamentos que cumplen con los criterios
"""

# %%
ncols = 3
nrows = int(np.ceil(len(final_results.groupby("localidad")) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 3))

for i, (l_dad, result) in enumerate(final_results.groupby("localidad")):
    plot_rt(result, axes.flat[i], l_dad)

fig.tight_layout()
fig.set_facecolor("w")
plt.savefig("gráficos/bta_localidades_rt.svg")

# %% [markdown]
"""
### Export Data to CSV
"""

# %%
# Uncomment the following line if you'd like to export the data
final_results.to_csv(f"datos/rt_bta_localidades.csv")


# %% [markdown]
"""
### Clasificaciones finales
"""

# %%
FULL_COLOR = [0.7, 0.7, 0.7]
NONE_COLOR = [179 / 255, 35 / 255, 14 / 255]
PARTIAL_COLOR = [0.5, 0.5, 0.5]
ERROR_BAR_COLOR = [0.3, 0.3, 0.3]


# %%
final_results


# %%
FILTERED_REGIONS = []
filtered = final_results.index.get_level_values(0).isin(FILTERED_REGIONS)
mr = final_results.loc[~filtered].groupby(level=0)[["ML", "High_90", "Low_90"]].last()


def plot_standings(mr, figsize=None, title="Most Likely Recent $R_t$ by County"):
    # if not figsize:
    #     figsize = ((15.9 / 50) * len(mr) + 0.1, 4.6)

    fig, ax = plt.subplots()

    ax.set_title(title)
    err = mr[["Low_90", "High_90"]].sub(mr["ML"], axis=0).abs()
    bars = ax.bar(
        mr.index,
        mr["ML"],
        width=0.825,
        color=FULL_COLOR,
        ecolor=ERROR_BAR_COLOR,
        capsize=2,
        error_kw={"alpha": 0.5, "lw": 1},
        yerr=err.values.T,
    )

    labels = mr.index.to_series()
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0, 2.0)
    ax.axhline(1.0, linestyle=":", color="k", lw=1)

    fig.tight_layout()
    fig.set_facecolor("w")
    return fig, ax


mr.sort_values("ML", inplace=True)
plot_standings(mr, title=f"Valores más probables de $R_t$ | {latest_date}")
plt.savefig("gráficos/bogota_ml_rt.svg")


# %%
mr.sort_values("High_90", inplace=True)
plot_standings(mr, title=f"Valores más (altos) probables de $R_t$ | {latest_date}")
("")
plt.savefig("gráficos/bogota_rt_alto.svg")


# %%
show = mr[mr.High_90.le(1)].sort_values("ML")
fig, ax = plot_standings(
    show, title=f"Localidades que tienen la \n pandemia bajo control \n {latest_date}"
)
plt.savefig("gráficos/bogota_rt_controlado.svg")


# %%
show = mr[mr.Low_90.ge(1.0)].sort_values("Low_90")
fig, ax = plot_standings(
    show,
    title=f"Localidades que no tienen \n la pandemia bajo control \n {latest_date}",
)
plt.savefig("gráficos/bogota_rt_descontrolada.svg")


# %%
cronologia = "datos/cronologia.csv"

bogota = pd.read_csv(cronologia)

# Print Counties
latest_date = bogota[-1:]
latest_date = latest_date.fecha
latest_date = " ".join(str(elem) for elem in latest_date)
print(latest_date)


ortografia = lambda error: error.replace(" Kennedy", "Kennedy")

bogota["localidad"] = bogota.apply(lambda x: ortografia(x["localidad"]), axis=1)

bog_latest = bogota
bog_latest.drop(bog_latest[bog_latest["fecha"] != latest_date].index, inplace=True)

bog_latest

pop_bogota = pd.read_csv("datos/bog_localidad.csv")
pop_bogota = pop_bogota.set_index(["localidad"]).sort_index()

rt_bogota = pd.read_csv("datos/rt_bta_localidades.csv")
latest_rt_bogota = rt_bogota[rt_bogota.fecha == latest_date]
localidades_rt = list(latest_rt_bogota["localidad"])


pop_bogota = pop_bogota[pop_bogota.index.isin(localidades_rt)]

bog_latest = bog_latest[bog_latest.localidad.isin(localidades_rt)]

bog_latest = bog_latest.sort_values(by=["localidad"])


bog_latest["población"] = list(pop_bogota["población"])
bog_latest["densidad_hab_km2"] = list(pop_bogota["densidad_hab_km2"])
bog_latest["tasa_casos_por_población"] = round(
    bog_latest["casos"] / bog_latest["población"], 4
)
bog_latest["ML"] = list(latest_rt_bogota["ML"])
bog_latest["codigo"] = bog_latest["codigo"].astype(int)
bog_latest.to_csv("datos/bog_latest.csv", index=False)
bog_latest.sort_values(by=["casos"], ascending=False)


# %%
import json
from urllib.request import urlopen

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
