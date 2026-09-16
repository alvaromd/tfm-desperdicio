# -*- coding: utf-8 -*-
"""
Prepara los datos que consume el cuadro de mando.

Entrena un modelo cuantilico por cada nivel del 0,05 al 0,95 y guarda su
prediccion para cada categoria y semana del periodo de prueba. Anade el
precio medio por unidad y el departamento de cada categoria, que son lo que
permite a la aplicacion razonar en euros y separar perecedero de no
perecedero.

El abanico es mas ancho que la frontera del capitulo 7 porque el producto no
perecedero pide coberturas muy altas.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEMILLA, CORTE = 42, 88
CUANTILES = [round(0.05 * k, 2) for k in range(1, 20)]   # del 0,05 al 0,95
PERECEDEROS = {"PRODUCE", "MEAT", "MEAT-PCKGD", "DELI",
               "PASTRY", "SEAFOOD", "SEAFOOD-PCKGD"}
PROC, CRUDO = Path("data/processed"), Path("data/raw")
SAL = Path("cuadro_mando/datos")
SAL.mkdir(parents=True, exist_ok=True)

panel = (pd.read_parquet(PROC / "panel_modelado.parquet")
           .sort_values(["COMMODITY_DESC", "WEEK_NO"]).reset_index(drop=True))
g = panel.groupby("COMMODITY_DESC", observed=True)
for k in [1, 2, 3, 4]:
    panel[f"lag_{k}"] = g.unidades.shift(k)
for w in [4, 8, 13]:
    panel[f"media_{w}"] = g.unidades.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
panel["desv_4"] = g.unidades.shift(1).rolling(4).std().reset_index(level=0, drop=True)
panel["tendencia_4"] = panel["media_4"] - panel["lag_1"]
panel["ratio_lag1_media4"] = panel["lag_1"] / panel["media_4"].replace(0, np.nan)
panel["nivel_categoria"] = g.unidades.transform(lambda s: s.shift(1).expanding().mean())
panel["semana_del_ano"] = panel.WEEK_NO % 52
panel["sin_ano"] = np.sin(2 * np.pi * panel.semana_del_ano / 52)
panel["cos_ano"] = np.cos(2 * np.pi * panel.semana_del_ano / 52)
panel["hogares_lag1"] = g.hogares_activos.shift(1)

FEATURES = [f"lag_{k}" for k in [1, 2, 3, 4]] + [f"media_{w}" for w in [4, 8, 13]] + \
           ["desv_4", "tendencia_4", "ratio_lag1_media4", "nivel_categoria",
            "sin_ano", "cos_ano", "hogares_lag1"]
datos = panel.dropna(subset=FEATURES + ["unidades"]).copy()
train, test = datos[datos.WEEK_NO < CORTE], datos[datos.WEEK_NO >= CORTE]
X_tr, y_tr, X_te = train[FEATURES].values, train.unidades.values, test[FEATURES].values
print(f"Entrenamiento {len(train):,} obs / prueba {len(test):,} obs")

salida = test[["COMMODITY_DESC", "WEEK_NO", "unidades"]].copy()
salida["ingenua"] = test["lag_1"].values          # politica de referencia
print(f"\nEntrenando {len(CUANTILES)} modelos cuantilicos")
for a in CUANTILES:
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=a,
                     n_estimators=600, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                     random_state=SEMILLA, n_jobs=-1, tree_method="hist")
    m.fit(X_tr, y_tr)
    salida[f"q{int(a * 100):02d}"] = np.clip(m.predict(X_te), 0, None)
    print(f"  cuantil {a:.2f} listo")

# --- atributos por categoria
precio = (panel.groupby("COMMODITY_DESC", observed=True)[["unidades", "ventas"]].sum()
               .assign(precio=lambda x: x.ventas / x.unidades).precio)
depart = (pd.read_csv(CRUDO / "product.csv", usecols=["DEPARTMENT", "COMMODITY_DESC"])
            .drop_duplicates("COMMODITY_DESC").set_index("COMMODITY_DESC").DEPARTMENT)

cat = pd.DataFrame({"precio": precio})
cat["departamento"] = cat.index.map(depart)
cat["perecedero"] = cat.departamento.isin(PERECEDEROS)
cat = cat.reset_index()

print(f"\nPrecio medio por unidad: mediana {precio.median():.2f} €, "
      f"de {precio.min():.2f} € a {precio.max():.2f} €")
print(f"Categorias perecederas: {int(cat.perecedero.sum())} de {len(cat)}")

salida.to_parquet(SAL / "predicciones_cuantiles.parquet", index=False)
cat.to_parquet(SAL / "categorias.parquet", index=False)
panel[["COMMODITY_DESC", "WEEK_NO", "unidades"]].to_parquet(SAL / "historico.parquet", index=False)

print(f"\nGuardado en {SAL}/")
for f in sorted(SAL.iterdir()):
    print(f"  {f.name:<34} {f.stat().st_size / 1024:7.1f} KB")
