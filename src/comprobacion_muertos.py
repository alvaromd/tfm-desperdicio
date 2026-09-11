# -*- coding: utf-8 -*-
"""
Comprobacion: los productos sin ventas recientes, estan realmente fuera del surtido
o simplemente el panel de hogares no captura sus ventas?

La respuesta cambia la interpretacion del resultado del reparto descendente.
"""
import pandas as pd
from pathlib import Path

RAW, PROC = Path("data/raw"), Path("data/processed")
CORTE = 88

trans = pd.read_csv(RAW / "transaction_data.csv",
                    usecols=["PRODUCT_ID", "QUANTITY", "WEEK_NO"])
prod = pd.read_csv(RAW / "product.csv", usecols=["PRODUCT_ID", "COMMODITY_DESC"])
cats = set(pd.read_parquet(PROC / "panel_modelado.parquet").COMMODITY_DESC.unique())

df = trans.merge(prod, on="PRODUCT_ID", how="left")
df = df[df.COMMODITY_DESC.isin(cats) & (df.WEEK_NO >= 16) &
        (df.QUANTITY > 0) & (df.QUANTITY <= 1000)]

sp = df.groupby(["PRODUCT_ID", "WEEK_NO"]).QUANTITY.sum().reset_index(name="uds")
train = sp[sp.WEEK_NO < CORTE]
test = sp[sp.WEEK_NO >= CORTE]

vivos = set(train.PRODUCT_ID)
recientes = set(train[train.WEEK_NO >= CORTE - 13].PRODUCT_ID)
sin_recientes = vivos - recientes          # "muertos" segun la ventana de 13 semanas
con_venta_test = set(test.PRODUCT_ID)

resucitan = sin_recientes & con_venta_test

print("=" * 66)
print("LOS PRODUCTOS SIN VENTAS RECIENTES, ESTAN MUERTOS DE VERDAD?")
print("=" * 66)
print(f"Productos con historia en entrenamiento          : {len(vivos):,}")
print(f"Sin ventas en las ultimas 13 semanas de train    : {len(sin_recientes):,} "
      f"({100*len(sin_recientes)/len(vivos):.1f}%)")
print(f"De esos, vuelven a vender en el periodo de prueba: {len(resucitan):,} "
      f"({100*len(resucitan)/len(sin_recientes):.1f}%)")

uds_res = test[test.PRODUCT_ID.isin(resucitan)].uds.sum()
print(f"\nUnidades que aportan los que resucitan: {uds_res:,.0f} "
      f"({100*uds_res/test.uds.sum():.1f}% de la demanda del periodo de prueba)")

# cuantas semanas vendian esos productos cuando estaban activos
act = train[train.PRODUCT_ID.isin(sin_recientes)].groupby("PRODUCT_ID").WEEK_NO.nunique()
print(f"\nCuando estaban activos, esos productos vendian en una mediana de "
      f"{act.median():.0f} semanas de las 72 del entrenamiento.")
print(f"El {100*(act <= 3).mean():.1f}% de ellos vendio en 3 semanas o menos en total.")

# contraste: el panel es una muestra
print("\n" + "=" * 66)
print("COBERTURA DEL PANEL A CADA NIVEL")
print("=" * 66)
sem = df.groupby("WEEK_NO").agg(prod_distintos=("PRODUCT_ID", "nunique"),
                                uds=("QUANTITY", "sum"))
print(f"Productos distintos vendidos por semana: mediana de {sem.prod_distintos.median():,.0f}")
print(f"Unidades por producto y semana         : mediana de "
      f"{sp.uds.median():.1f}, media de {sp.uds.mean():.1f}")

panel = pd.read_parquet(PROC / "panel_modelado.parquet")
print(f"\nA nivel de CATEGORIA, en cambio:")
print(f"  observaciones con demanda nula: {(panel.unidades == 0).sum()} de {len(panel):,}")
print(f"  unidades por categoria y semana: mediana de {panel.unidades.median():,.0f}")
