"""
EDA del dataset Dunnhumby "The Complete Journey".
Objetivo: caracterizar la demanda semanal por categoria para construir el target
proxy de sobrestock, y detectar productos/categorias de baja rotacion.

Produce un resumen numerico por consola y los paneles agregados en
data/processed. Las figuras definitivas las genera src/figuras_memoria.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)


def seccion(titulo):
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


# ---------------------------------------------------------------- carga
seccion("1. CARGA DE DATOS")

trans = pd.read_csv(
    RAW / "transaction_data.csv",
    usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY",
             "SALES_VALUE", "STORE_ID", "WEEK_NO", "RETAIL_DISC", "COUPON_DISC"],
)
prod = pd.read_csv(
    RAW / "product.csv",
    usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"],
)

print(f"transacciones : {len(trans):>10,} filas")
print(f"productos     : {len(prod):>10,} filas")
print(f"hogares       : {trans.household_key.nunique():>10,}")
print(f"tiendas       : {trans.STORE_ID.nunique():>10,}")
print(f"semanas       : {trans.WEEK_NO.min()} a {trans.WEEK_NO.max()} "
      f"({trans.WEEK_NO.nunique()} semanas ~ {trans.WEEK_NO.nunique()/52:.1f} anios)")
print(f"dias          : {trans.DAY.min()} a {trans.DAY.max()}")

# ------------------------------------------------- calidad de los datos
seccion("2. CALIDAD DE LOS DATOS")

print("Nulos por columna (transacciones):")
nulos = trans.isna().sum()
print(nulos[nulos > 0] if nulos.sum() else "  sin nulos")

print(f"\nQUANTITY: min={trans.QUANTITY.min()}, max={trans.QUANTITY.max():,}, "
      f"mediana={trans.QUANTITY.median()}")
print(f"  cantidad = 0      : {(trans.QUANTITY == 0).sum():,} filas "
      f"({(trans.QUANTITY == 0).mean()*100:.2f}%)")
print(f"  cantidad > 100    : {(trans.QUANTITY > 100).sum():,} filas "
      f"(probable producto a granel/peso)")
print(f"\nSALES_VALUE: min={trans.SALES_VALUE.min()}, max={trans.SALES_VALUE.max():,.2f}, "
      f"mediana={trans.SALES_VALUE.median():.2f}")
print(f"  importe <= 0      : {(trans.SALES_VALUE <= 0).sum():,} filas "
      f"(devoluciones o promociones al 100%)")

# productos vendidos que no estan en el catalogo
sin_catalogo = ~trans.PRODUCT_ID.isin(prod.PRODUCT_ID)
print(f"\nTransacciones de productos ausentes del catalogo: {sin_catalogo.sum():,}")

# ----------------------------------------------- union con el catalogo
seccion("3. ESTRUCTURA DE CATEGORIAS")

df = trans.merge(prod, on="PRODUCT_ID", how="left")
df["DEPARTMENT"] = df["DEPARTMENT"].fillna("SIN DEPARTAMENTO")
df["COMMODITY_DESC"] = df["COMMODITY_DESC"].fillna("SIN CATEGORIA")

print(f"departamentos            : {df.DEPARTMENT.nunique()}")
print(f"categorias (COMMODITY)   : {df.COMMODITY_DESC.nunique()}")
print(f"subcategorias            : {df.SUB_COMMODITY_DESC.nunique()}")

top_dep = (df.groupby("DEPARTMENT")
             .agg(ventas=("SALES_VALUE", "sum"), lineas=("SALES_VALUE", "size"))
             .sort_values("ventas", ascending=False))
top_dep["cuota_%"] = 100 * top_dep.ventas / top_dep.ventas.sum()
print("\nTop 10 departamentos por facturacion:")
print(top_dep.head(10).to_string(float_format=lambda x: f"{x:,.1f}"))

# --------------------------------------- serie temporal agregada global
seccion("4. EVOLUCION TEMPORAL Y ESTACIONALIDAD")

sem = (df.groupby("WEEK_NO")
         .agg(unidades=("QUANTITY", "sum"),
              ventas=("SALES_VALUE", "sum"),
              cestas=("BASKET_ID", "nunique"),
              hogares=("household_key", "nunique"))
         .reset_index())

print(f"Ventas semanales: media={sem.ventas.mean():,.0f}, "
      f"cv={sem.ventas.std()/sem.ventas.mean():.3f}")
print("\nPrimeras y ultimas 3 semanas (deteccion de bordes incompletos):")
print(pd.concat([sem.head(3), sem.tail(3)]).to_string(index=False,
      float_format=lambda x: f"{x:,.0f}"))


# ------------------------------- panel semana x categoria (el target)
seccion("5. PANEL SEMANA x CATEGORIA (base del target)")

panel = (df.groupby(["WEEK_NO", "COMMODITY_DESC"])
           .agg(unidades=("QUANTITY", "sum"),
                ventas=("SALES_VALUE", "sum"),
                cestas=("BASKET_ID", "nunique"),
                hogares=("household_key", "nunique"))
           .reset_index())

print(f"Panel resultante: {len(panel):,} filas "
      f"({panel.WEEK_NO.nunique()} semanas x {panel.COMMODITY_DESC.nunique()} categorias)")

# cobertura: en cuantas semanas aparece cada categoria
cobertura = (panel.groupby("COMMODITY_DESC").WEEK_NO.nunique()
             .rename("semanas_activas").reset_index())
cobertura["cobertura_%"] = 100 * cobertura.semanas_activas / panel.WEEK_NO.nunique()

completas = cobertura["cobertura_%"]
print(f"\nCategorias presentes en el 100% de las semanas: "
      f"{(completas == 100).sum()} de {len(cobertura)}")
print(f"Categorias presentes en <50% de las semanas    : {(completas < 50).sum()}")

vol_cat = (panel.groupby("COMMODITY_DESC")
             .agg(ventas_tot=("ventas", "sum"), unidades_tot=("unidades", "sum"))
             .sort_values("ventas_tot", ascending=False))
vol_cat["cuota_%"] = 100 * vol_cat.ventas_tot / vol_cat.ventas_tot.sum()
vol_cat["cuota_acum_%"] = vol_cat["cuota_%"].cumsum()

print("\nTop 15 categorias por facturacion:")
print(vol_cat.head(15).to_string(float_format=lambda x: f"{x:,.1f}"))

n80 = int(vol_cat["cuota_acum_%"].searchsorted(80)) + 1
print(f"\nConcentracion: {n80} categorias concentran el 80% de la facturacion "
      f"({100*n80/len(vol_cat):.1f}% del total de categorias)")

# ------------------------------------- volatilidad de la demanda
seccion("6. VOLATILIDAD DE LA DEMANDA POR CATEGORIA")

# solo categorias con presencia completa, que son las modelables
cats_ok = cobertura.loc[completas == 100, "COMMODITY_DESC"]
p_ok = panel[panel.COMMODITY_DESC.isin(cats_ok)]

vol = (p_ok.groupby("COMMODITY_DESC").unidades
         .agg(media="mean", sd="std")
         .assign(cv=lambda d: d.sd / d.media)
         .sort_values("cv", ascending=False))

print(f"Categorias analizadas (presencia completa): {len(vol)}")
print(f"Coeficiente de variacion: mediana={vol.cv.median():.3f}, "
      f"p90={vol.cv.quantile(.9):.3f}")
print("\nTop 10 categorias MAS volatiles (mayor riesgo de desajuste stock/demanda):")
print(vol.head(10).to_string(float_format=lambda x: f"{x:,.2f}"))
print("\nTop 10 categorias MAS estables (demanda predecible):")
print(vol.tail(10).to_string(float_format=lambda x: f"{x:,.2f}"))


# ---------------------------------------------- rotacion de productos
seccion("7. ROTACION DE PRODUCTOS (senal de desperdicio)")

rot = (df.groupby("PRODUCT_ID")
         .agg(unidades=("QUANTITY", "sum"),
              semanas_con_venta=("WEEK_NO", "nunique"),
              ventas=("SALES_VALUE", "sum")))
n_sem = df.WEEK_NO.nunique()

print(f"Productos con al menos una venta: {len(rot):,} de {len(prod):,} del catalogo")
print(f"  vendidos en 1 sola semana : {(rot.semanas_con_venta == 1).sum():,} "
      f"({(rot.semanas_con_venta == 1).mean()*100:.1f}%)")
print(f"  vendidos en <10% semanas  : {(rot.semanas_con_venta < n_sem*.1).sum():,} "
      f"({(rot.semanas_con_venta < n_sem*.1).mean()*100:.1f}%)")
print(f"  vendidos en >90% semanas  : {(rot.semanas_con_venta > n_sem*.9).sum():,} "
      f"({(rot.semanas_con_venta > n_sem*.9).mean()*100:.1f}%)")

rot_sorted = rot.sort_values("ventas", ascending=False)
cuota = 100 * rot_sorted.ventas.cumsum() / rot_sorted.ventas.sum()
n_prod_80 = int((cuota <= 80).sum()) + 1
print(f"\nConcentracion: {n_prod_80:,} productos ({100*n_prod_80/len(rot):.1f}%) "
      f"generan el 80% de la facturacion. Cola larga muy marcada.")


# ------------------------------------------------ guardar el panel
panel.to_parquet(PROC / "panel_semana_categoria.parquet", index=False)
sem.to_parquet(PROC / "serie_semanal_global.parquet", index=False)
vol.to_csv(PROC / "volatilidad_categorias.csv")
vol_cat.to_csv(PROC / "volumen_categorias.csv")

seccion("RESUMEN")
print(f"Panel guardado          : data/processed/panel_semana_categoria.parquet")
print(f"Categorias modelables   : {len(vol)} (presencia en las 102 semanas)")
print(f"Observaciones del panel  : {len(p_ok):,} para modelado")
