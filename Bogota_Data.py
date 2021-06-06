# -*- coding: utf-8 -*-
# %%
# import chardet
# import subprocess

import pandas as pd
from IPython import get_ipython

# %% [markdown]
# # Bogotá Data


# %%
# Comando para descargar los datos de la página del Distrito en una carpeta llamada 'datos'

subprocess.run(
    [
        "wget",
        "https://datosabiertos.bogota.gov.co/dataset/44eacdb7-a535-45ed-be03-16dbbea6f6da/resource/b64ba3c4-9e41-41b8-b3fd-2da21d627558/download/osb_enftransm-covid-19.csv",
        "-P",
        "datos/",
    ]
)


# %%
url = "https://datosabiertos.bogota.gov.co/api/3/action/datastore_search?resource_id=b64ba3c4-9e41-41b8-b3fd-2da21d627558&limit=5"


# %%
import json
from urllib.request import urlopen

with urlopen(url) as response:
    test = json.load(response)


test.keys()
# basePath = test.get("basePath")
# csvPath = test.get("regions").get("en-us").get("csvPath")


# %%
# test.get("result").get

# %%
json = pd.DataFrame(test.get("result").get("records"))


# %%
cols = list(json.columns)


# %%
cols.remove("_id")


# %%
len(cols)


# %%
cols


# %%
import chardet

# look at the first ten thousand bytes to guess the character encoding
with open("datos/osb_enftransm-covid-19.csv", "rb",) as rawdata:
    result = chardet.detect(rawdata.read(1000))

# check what the character encoding might be
print(result)

# %%
print(result["encoding"])

# %%
# El equipo de Datos Abiertos de Bogotá, que primero, que tesos por tener la
# plataforma de datos para hacer todo esto posible, por alguna razón, no han
# podido ponerse de acuerdo si un día van a generar el archivo .CSV delimitado
# por comas (,) o Punto y comas (;). No me dejaron opción sino crear la
# condición para ambos.

# try:
#     bogota = pd.read_csv(
#         "datos/osb_enftransm-covid-19.csv",
#         encoding="iso-8859-1",
#         cache_dates=True,
#         parse_dates=[s for s in cols if "DIAGNOSTICO" in s],
#         dayfirst=True,
#         sep=",",
#         #         usecols=cols,
#     )
# except ValueError:

bogota = pd.read_csv(
    "datos/osb_enftransm-covid-19.csv",
#     encoding="ascii",
    cache_dates=True,
    parse_dates=[s for s in cols if "DIAGNOSTICO" in s],
    dayfirst=True,
    sep=";",
    skipfooter=5,
    encoding=result["encoding"],
    engine="python",
    # converters={
    #     "LOCALIDAD_ASIS": lambda x: x.replace("Antonio Nari�o", "Antonio Nariño"),
    #     "LOCALIDAD_ASIS": lambda x: x.repace("Engativ�": "Engativá")
    # }
    #         usecols=cols,
)

bogota = bogota.dropna(how="all")
print(set(bogota["LOCALIDAD_ASIS"]))

# %%
desorganizacion = sorted(set(bogota["LOCALIDAD_ASIS"]))
organizacion = [
    "Antonio Nariño",
    "Barrios Unidos",
    "Bosa",
    "Chapinero",
    "Ciudad Bolívar",
    "Engativá",
    "Fontibón",
    "Fuera de Bogotá",
    "Kennedy",
    "La Candelaria",
    "Los Mártires",
    "Puente Aranda",
    "Rafael Uribe Uribe",
    "San Cristóbal",
    "Santa Fe",
    "Sin dato",
    "Suba",
    "Sumapaz",
    "Teusaquillo",
    "Tunjuelito",
    "Usaquén",
    "Usme",
]

# %%
localidades = dict(zip(desorganizacion, organizacion))

# %%
if set(bogota["LOCALIDAD_ASIS"]) != set(localidades.values()):
    bogota["LOCALIDAD_ASIS"] = bogota["LOCALIDAD_ASIS"].map(localidades)

# %%
print(set(bogota["LOCALIDAD_ASIS"]))

# %%
import os, glob
for filename in glob.glob("datos/osb*"):
    os.remove(filename) 

# %%
bogota.head()

# %%
bogota.tail()


# %%
bogota.info()


# %%
bogota = bogota[
    [
        [i for i in cols if i.startswith("FECHA")][0],
        [i for i in cols if i.startswith("LOCAL")][0],
        [i for i in cols if i.startswith("EDAD")][0],
        [i for i in cols if i.startswith("UNI")][0],
        [i for i in cols if i.startswith("SEX")][0],
        [i for i in cols if i.startswith("FUEN")][0],
        [i for i in cols if i.startswith("UBI")][0],
        [i for i in cols if i.startswith("ES")][0],
    ]
]

# %% [markdown]
"""
Cabe resaltar que hay dos sujetos de la información que están mal digitados con
más de los 8 atributos creados en la tabla. Tuve que borrar los detalles de la
tabla para pudiera ser importada.

**Nota**: Unidad de medidad de la edad: `1`= años, `2`= meses, `3`= días.
"""

# %%
bogota


# %%
# Vamos a simplificar el nombre de las columnas para poder manipularlas con más
# facilidad

bogota.columns = [
    "fecha",
    "localidad",
    "edad",
    "unidad_edad",
    "sexo",
    "tipo_de_caso",
    "ubicación",
    "estado",
]

bogota


# %%
bogota = bogota[bogota["fecha"].notna()]


# %%
bogota


# %%
bogota = bogota.drop(bogota[bogota.fecha == "#ĦREF!"].index)


# %%
def fix_all_dates(df, date):
    """
    Función para arreglar las fechas que están en formato Excel como números
    (por ejemplo. hay fechas como `44205`) cuando a la vez hay fechas en la
    columna en formato estándar.
    """

    from datetime import datetime

    # establecer el tipo de columna como una `string` si aún no lo está
    df[date] = df[date].astype("str")

    # crea una máscara de fecha basada en la cadena que contiene un /
    date_mask = df[date].str.len() != 5

    # dividir las fechas para excel
    df_excel = df[~date_mask].copy()

    # dividir las fechas regulares
    df_reg = df[date_mask].copy()

    # convertir fechas de registro a fecha y hora
    # df_reg[date] = pd.to_datetime(df_reg[date], format="%d/%m/%Y")
    df_reg[date] = pd.to_datetime(df_reg[date])

    # convertir fechas de Excel a fecha y hora;
    # la columna debe convertirse en `ints`
    df_excel[date] = pd.TimedeltaIndex(df_excel[date].astype(int), unit="d") + datetime(
        1899, 12, 30
    )

    # combinar los dataframes
    df = pd.concat([df_reg, df_excel])

    return df


# %%
bogota = fix_all_dates(bogota, "fecha")

# %%
bogota.tail()


# %%
# bogota["fecha"] = bogota["fecha"].apply(pd.to_datetime, format="%d/%m/%Y")


# %%
bogota = bogota.sort_values("fecha")


# %%
bogota

# %%
print(bogota.columns)


# %%
# Por alguna razón, las localidades aparecen con un espacio de más, por ejemplo:
# " Kennedy" en vez de "Kennedy", aquí creamos una función que eliminará
# cualquier espacio sobrante a la izquierda
# ortografia = lambda error: error.lstrip()


# %%
bogota = bogota.dropna()


# %%
# Aquí estamos creando una lista con las localidades de la tabla

localidades = list(bogota["localidad"].unique())
print(localidades)


# %%
# Aqui esta un conteo de los casos diarios de la locallidad de Kennedy, se puede
# insertar cualquiera de las localidades para encontrar conteos.

# bogota.loc[(bogota.localidad == "Kennedy"), "fecha"].value_counts(
#     dropna=False
# ).sort_values("fecha").cumsum(skipna=False)


# %%
# Aquí vamos a crear una lista de las fechas
fechas = list(bogota["fecha"].unique())
print(fechas[-1])


# %%
bogota_c = pd.DataFrame()

# with open("datos/bog_output.txt", "w") as text_file:
for l in localidades:
    conteo = bogota.loc[(bogota.localidad == l), "fecha"].value_counts()
    conteo = conteo.sort_index(ascending=True)
    conteo = conteo.cumsum()
    # print(
    #     l, "\n", conteo, sep="", file=text_file,
    # )
    bogota_c = bogota_c.append(conteo)


# %%
bogota_c = bogota_c.T


# %%
bogota_c.columns = localidades

# %% [markdown]
"""
From fillna method descriptions:

```python
method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None Method to use for filling holes in reindexed Series pad / ffill:

# propagate last valid observation forward to next valid backfill / bfill: use NEXT valid observation to fill gap

df_new.Inventory = df_new.Inventory.fillna(method="ffill")
```
"""

# %%
bogota_c = bogota_c.sort_index(axis=0)
bogota_c = bogota_c.fillna(method="ffill")


# %%
bogota_c.info()


# %%
bogota_c

# %%
# def codigo(ldad):
#     """Función para crear columna del código de localidad"""
#     codigo = {
#         "Usaquén": 1,
#         "Chapinero": 2,
#         "Santa Fe": 3,
#         "San Cristóbal": 4,
#         "Usme": 5,
#         "Tunjuelito": 6,
#         "Bosa": 7,
#         "Kennedy": 8,
#         "Fontibón": 9,
#         "Engativá": 10,
#         "Suba": 11,
#         "Barrios Unidos": 12,
#         "Teusaquillo": 13,
#         "Los Mártires": 14,
#         "Antonio Nariño": 15,
#         "Puente Aranda": 16,
#         "La Candelaria": 17,
#         "Rafael Uribe Uribe": 18,
#         "Ciudad Bolívar": 19,
#         "Sumapaz": 20,
#         "Fuera de Bogotá": 99,
#         "Sin dato": 99,
#     }

#     return codigo.get(ldad)


# codigo("Puente Aranda")


# %%
codigo = {
    "Usaquén": 1,
    "Chapinero": 2,
    "Santa Fe": 3,
    "San Cristóbal": 4,
    "Usme": 5,
    "Tunjuelito": 6,
    "Bosa": 7,
    "Kennedy": 8,
    "Fontibón": 9,
    "Engativá": 10,
    "Suba": 11,
    "Barrios Unidos": 12,
    "Teusaquillo": 13,
    "Los Mártires": 14,
    "Antonio Nariño": 15,
    "Puente Aranda": 16,
    "La Candelaria": 17,
    "Rafael Uribe Uribe": 18,
    "Ciudad Bolívar": 19,
    "Sumapaz": 20,
    "Fuera de Bogotá": 99,
    "Sin dato": 99,
}


# %%
x = bogota_c
y = x.reset_index()
z = pd.melt(y, id_vars=["index"], value_vars=localidades,)
z.columns = ["fecha", "localidad", "casos"]
z = z.sort_values(by=["fecha", "casos"])
z = z.dropna()
# z["codigo"] = z.apply(lambda x: codigo(x["localidad"]), axis=1)
z["codigo"] = z["localidad"].map(codigo)
z
cronologia = z


# %%
cronologia


# %%
cronologia = cronologia.query("localidad != 'Sin dato'")


# %%
cronologia


# %%
cronologia.to_csv("datos/cronologia.csv", index=False)


# %%
bogota["fecha"].value_counts(dropna=False).head(10)


# %%
bogota = bogota[["fecha"]].reset_index(drop=True)
bogota.tail(10)


# %%
import datetime

fixed_dates_df = bogota.copy()
fixed_dates_df["fecha"] = fixed_dates_df["fecha"].apply(pd.to_datetime)
fixed_dates_df = fixed_dates_df.set_index(fixed_dates_df["fecha"])
grouped = fixed_dates_df.resample("D").count()
data_df = pd.DataFrame({"count": grouped.values.flatten()}, index=grouped.index)
data_df.tail(10)


# %%
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
plt.style.use("ggplot")

data_df.plot(color="purple")


# %%
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data_df)
fig = result.plot()
fig.tight_layout()


# %%
data_df.info()


# %%
from fbprophet import Prophet

model = Prophet(daily_seasonality=True)
train_df = data_df.rename(columns={"count": "y"})
train_df["ds"] = train_df.index
model.fit(train_df)


# %%
pd.plotting.register_matplotlib_converters()
future = model.make_future_dataframe(365, freq="D", include_history=True)
forecast = model.predict(future)
model.plot(forecast)
