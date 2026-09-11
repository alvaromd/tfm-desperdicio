# -*- coding: utf-8 -*-
"""
Puntos de referencia para la frontera del cuadro de mando.

El capitulo 6 traza la frontera aplicando un colchon de seguridad del 10 %
sobre cada prediccion. El cuadro de mando lo retiro, porque sumar un
colchon a una prediccion de cuantil duplica el margen de seguridad: el
cuantil ya codifica el nivel de servicio buscado.

Eso hace que las cifras absolutas del cuadro de mando no coincidan con las
de la figura 16, aunque la historia sea la misma. Para que dentro de la
aplicacion todo sea comparable, aqui se recalculan las tres politicas de
referencia con la misma regla que usa el cuadro de mando, es decir sin
colchon: reponer exactamente lo que dice la politica.

  - Ingenua: repetir lo vendido la semana pasada.
  - Media movil de 4: la media de las cuatro semanas anteriores.
  - XGBoost simetrico: el modelo entrenado con error cuadratico, que
    predice la demanda esperada y no un cuantil de ella.

Las tres deberian quedar por encima de la frontera cuantilica en el plano
excedente-rotura, que es el argumento del trabajo: no basta con predecir
bien la media, hay que predecir la cantidad que conviene reponer.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RES, SAL = Path("results"), Path("cuadro_mando/datos")

t = pd.read_parquet(RES / "predicciones_test.parquet")
real = t.unidades.values
demanda = real.sum()

POLITICAS = [
    ("Repetir la semana pasada", t.lag_1.values),
    ("Media móvil de 4 semanas", t.media_4.values),
    ("XGBoost sin asimetría", t.pred.values),
]

filas = []
for nombre, pred in POLITICAS:
    rep = np.clip(pred, 0, None)
    exc = float(np.clip(rep - real, 0, None).sum())
    rot = float(np.clip(real - rep, 0, None).sum())
    filas.append({"modelo": nombre, "excedente": exc, "rotura": rot,
                  "exc_pct": 100 * exc / demanda, "rot_pct": 100 * rot / demanda})

ref = pd.DataFrame(filas)
ref.to_parquet(SAL / "referencias.parquet", index=False)

print(f"Demanda del periodo de prueba: {demanda:,.0f} unidades")
print(f"Filas: {len(t):,}\n")
print(ref.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print(f"\nGuardado en {SAL / 'referencias.parquet'}")
