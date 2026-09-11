"""
Profundizacion en tres hallazgos criticos del EDA inicial:
  A) Rampa de incorporacion de hogares al panel (bordes no representativos)
  B) Outliers extremos en QUANTITY y la categoria COUPON/MISC ITEMS
  C) Definicion del periodo y universo validos para el modelado
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW, PROC = Path("data/raw"), Path("data/processed")


def seccion(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


trans = pd.read_csv(RAW / "transaction_data.csv",
                    usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID",
                             "QUANTITY", "SALES_VALUE", "WEEK_NO"])
prod = pd.read_csv(RAW / "product.csv",
                   usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"])

# ------------------------------------------------------------------ A
seccion("A. RAMPA DE INCORPORACION DE HOGARES")

h = trans.groupby("WEEK_NO").household_key.nunique()
print("Hogares activos por semana (muestreo):")
for w in [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 50, 70, 90, 102]:
    print(f"  semana {w:>3}: {h.get(w, 0):>5,} hogares")

meseta = h[h.index >= 30].median()
print(f"\nMediana de hogares activos a partir de la semana 30: {meseta:,.0f}")
umbral = 0.9 * meseta
primera_ok = int(h[h >= umbral].index.min())
print(f"Primera semana que alcanza el 90% de la meseta ({umbral:,.0f}): {primera_ok}")

# ------------------------------------------------------------------ B
seccion("B. OUTLIERS Y CATEGORIAS ARTEFACTO")

df = trans.merge(prod, on="PRODUCT_ID", how="left")

print("Distribucion de QUANTITY:")
for q in [.5, .9, .99, .999, .9999, 1.0]:
    print(f"  percentil {q*100:>7.2f}: {df.QUANTITY.quantile(q):>15,.0f}")

extremos = df[df.QUANTITY > 1000]
print(f"\nLineas con QUANTITY > 1.000: {len(extremos):,} "
      f"({len(extremos)/len(df)*100:.4f}% de las lineas)")
print("Concentracion de esas lineas por categoria:")
print(extremos.COMMODITY_DESC.value_counts().head(8).to_string())

cm = df[df.COMMODITY_DESC == "COUPON/MISC ITEMS"]
print(f"\nCOUPON/MISC ITEMS: {len(cm):,} lineas, "
      f"{cm.QUANTITY.sum():,} unidades, {cm.SALES_VALUE.sum():,.0f} de facturacion")
print(f"  QUANTITY mediana={cm.QUANTITY.median():,.0f}, max={cm.QUANTITY.max():,.0f}")
print("  -> artefacto contable (cupones/varios), NO es mercancia perecedera")

print("\nDepartamentos que NO son alimentacion vendible y conviene excluir:")
no_alim = ["KIOSK-GAS", "MISC SALES TRAN", "COUP/STR & MFG", "GRO BAKERY",
           "PHOTO", "VIDEO", "VIDEO RENTAL", "POSTAL CENTER", "GM MERCH EXP",
           "CNTRL/STORE SUP", "DAIRY DELI", "CHARITABLE CONT", "TRAVEL & LEISUR",
           "ELECT &PLUMBING", "AUTOMOTIVE", "RX", "PROD-WHS SALES"]
presentes = [d for d in no_alim if d in df.DEPARTMENT.unique()]
print(" ", presentes)

# ------------------------------------------------------------------ C
seccion("C. UNIVERSO VALIDO PARA EL MODELADO")

# departamentos de alimentacion perecedera y de gran consumo
alim = ["GROCERY", "PRODUCE", "MEAT", "MEAT-PCKGD", "DELI", "PASTRY",
        "SEAFOOD", "SEAFOOD-PCKGD", "FROZEN GROCERY", "NUTRITION"]
alim = [d for d in alim if d in df.DEPARTMENT.unique()]
print(f"Departamentos retenidos: {alim}")

val = df[(df.DEPARTMENT.isin(alim)) &
         (df.COMMODITY_DESC != "COUPON/MISC ITEMS") &
         (df.WEEK_NO >= primera_ok) &
         (df.QUANTITY > 0) & (df.QUANTITY <= 1000) &
         (df.SALES_VALUE > 0)].copy()

print(f"\nFiltrado aplicado:")
print(f"  lineas totales          : {len(df):>10,}")
print(f"  lineas retenidas        : {len(val):>10,} ({len(val)/len(df)*100:.1f}%)")
print(f"  semanas                 : {val.WEEK_NO.min()} a {val.WEEK_NO.max()} "
      f"({val.WEEK_NO.nunique()} semanas)")
print(f"  categorias              : {val.COMMODITY_DESC.nunique()}")

panel = (val.groupby(["WEEK_NO", "COMMODITY_DESC"])
            .agg(unidades=("QUANTITY", "sum"),
                 ventas=("SALES_VALUE", "sum"),
                 cestas=("BASKET_ID", "nunique"),
                 hogares=("household_key", "nunique"))
            .reset_index())

n_sem = panel.WEEK_NO.nunique()
cob = panel.groupby("COMMODITY_DESC").WEEK_NO.nunique()
cats_full = cob[cob == n_sem].index
panel_f = panel[panel.COMMODITY_DESC.isin(cats_full)].copy()

print(f"\nPanel final: {len(panel_f):,} observaciones "
      f"({n_sem} semanas x {len(cats_full)} categorias con serie completa)")

# normalizacion por hogares activos: separa demanda real de tamanio del panel
act = val.groupby("WEEK_NO").household_key.nunique().rename("hogares_activos")
panel_f = panel_f.merge(act, on="WEEK_NO")
panel_f["unidades_por_hogar"] = panel_f.unidades / panel_f.hogares_activos

cv_bruto = (panel_f.groupby("COMMODITY_DESC").unidades
            .agg(lambda s: s.std() / s.mean()).median())
cv_norm = (panel_f.groupby("COMMODITY_DESC").unidades_por_hogar
           .agg(lambda s: s.std() / s.mean()).median())
print(f"\nCV mediano sin normalizar        : {cv_bruto:.3f}")
print(f"CV mediano normalizado por hogar : {cv_norm:.3f}")

panel_f.to_parquet(PROC / "panel_modelado.parquet", index=False)

seccion("CONCLUSION")
print(f"Periodo valido      : semanas {primera_ok} a {int(panel_f.WEEK_NO.max())}")
print(f"Categorias          : {len(cats_full)}")
print(f"Observaciones       : {len(panel_f):,}")
print(f"Panel guardado en   : data/processed/panel_modelado.parquet")
