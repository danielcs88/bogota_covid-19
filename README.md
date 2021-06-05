<meta name="viewport" content="width=device-width, initial-scale=1.0">

# Análisis de la pandemia COVID-19 en 🇨🇴 Bogotá a nivel de localidad | R<sub>t</sub>

por Daniel Cárdenas

Si R<sub>t</sub> > 1, el número de casos aumentará, como al comienzo de una
epidemia. Cuando R<sub>t</sub> = 1, la enfermedad es endémica, y cuando
R<sub>t</sub> < 1 habrá una disminución en el número de casos.

Entonces, los epidemiólogos usan R<sub>t</sub> para hacer recomendaciones de
políticas. Es por eso que este número es tan importante.

## Notas

Hay varios registros de infectados en el datos del Distrito donde no tienen
datos o la persona infectada no es reside en Bogotá.

Mi modelo es una adaptación del model de
[Kevin Systrom](https://github.com/k-sys/covid-19)

## Fuente de Datos

Mi fuente de datos es la plataforma de Datos Abiertos de Bogotá.

- [Número de casos confirmados por el laboratorio de COVID- 19 - Bogotá D.C.](https://datosabiertos.bogota.gov.co/dataset/numero-de-casos-confirmados-por-el-laboratorio-de-covid-19-bogota-d-c)

### Casos y Promedios Semanales

![](gráficos/Suba.svg)
![](gráficos/Kennedy.svg)
![](gráficos/Engativá.svg)
![](gráficos/Ciudad%20Bolívar.svg)
![](gráficos/Bosa.svg)
![](gráficos/Usaquén.svg)
![](gráficos/Usme.svg)
![](gráficos/San%20Cristóbal.svg)
![](gráficos/Fontibón.svg)
![](gráficos/Rafael%20Uribe%20Uribe.svg)
![](gráficos/Puente%20Aranda.svg)
![](gráficos/Barrios%20Unidos.svg)
![](gráficos/Tunjuelito.svg)
![](gráficos/Chapinero.svg)
![](gráficos/Santa%20Fe.svg)
![](gráficos/Antonio%20Nariño.svg)
![](gráficos/Los%20Mártires.svg)

### R<sub>t</sub> de localidades en Bogotá

![](gráficos/bta_localidades_rt.svg)

![](gráficos/bogota_ml_rt.svg)

![](gráficos/bogota_rt_controlado.svg)

![](gráficos/bogota_rt_descontrolada.svg)
