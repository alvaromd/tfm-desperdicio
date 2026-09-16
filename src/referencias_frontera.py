# -*- coding: utf-8 -*-
"""
Puntos de referencia para la frontera del cuadro de mando.

Recalcula tres politicas que no eligen cuantil (repetir la semana anterior,
media movil de cuatro semanas y XGBoost simetrico) con la misma regla que usa
el cuadro de mando, es decir sin el colchon del 10 % del capitulo 7. Solo asi
son comparables con la frontera que dibuja la aplicacion.
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
