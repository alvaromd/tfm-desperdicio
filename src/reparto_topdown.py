# -*- coding: utf-8 -*-
"""
Reparto descendente (top-down) de la prediccion de categoria a producto.

Contrasta empiricamente la afirmacion del estado del arte: predecir a nivel de
categoria y repartir despues entre las referencias mediante pesos historicos es
mas robusto que predecir cada referencia por separado.

Comparacion:
  - Descendente : prediccion de la categoria x peso historico del producto
  - Ascendente  : prediccion directa sobre la serie de cada producto
                  (ultimo valor, media movil de 4 semanas y media historica)

Los pesos se estiman UNICAMENTE con el tramo de entrenamiento, de modo que no se
filtra informacion del periodo de prueba.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error

RAW, PROC, FIG, RES = Path("data/raw"), Path("data/processed"), Path("figures"), Path("results")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
AZUL, NARANJA, VERDE, ROJO = "#2A6F97", "#E07A5F", "#4A7C59", "#C1121F"
CORTE = 88


def sec(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


# ---------------------------------------------------- 1. universo comun
sec("1. UNIVERSO DE TRABAJO")

trans = pd.read_csv(RAW / "transaction_data.csv",
                    usecols=["PRODUCT_ID", "QUANTITY", "SALES_VALUE", "WEEK_NO"])
prod = pd.read_csv(RAW / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"])

panel_cat = pd.read_parquet(PROC / "panel_modelado.parquet")
categorias = set(panel_cat.COMMODITY_DESC.unique())

df = trans.merge(prod, on="PRODUCT_ID", how="left")
df = df[df.COMMODITY_DESC.isin(categorias) & (df.WEEK_NO >= 16) &
        (df.QUANTITY > 0) & (df.QUANTITY <= 1000) & (df.SALES_VALUE > 0)].copy()

print(f"Categorias del panel      : {len(categorias)}")
print(f"Lineas del universo       : {len(df):,}")
print(f"Productos distintos       : {df.PRODUCT_ID.nunique():,}")

# serie producto x semana
sp = (df.groupby(["PRODUCT_ID", "COMMODITY_DESC", "WEEK_NO"], observed=True)
        .QUANTITY.sum().reset_index(name="unidades"))

train_p = sp[sp.WEEK_NO < CORTE]
# solo referencias con historia en entrenamiento: sin ella no hay peso que aplicar
vivos = train_p.PRODUCT_ID.unique()
print(f"Productos con historia en entrenamiento: {len(vivos):,}")

# --------------------------------------- 2. rejilla completa de evaluacion
sec("2. REJILLA DE EVALUACION")
print("Se evalua sobre la rejilla completa producto x semana del periodo de prueba.")
print("Las combinaciones sin venta se tratan como demanda nula, que es lo correcto:")
print("una referencia que no vende esa semana tiene demanda cero, no ausencia de dato.\n")

semanas_test = sorted(sp[sp.WEEK_NO >= CORTE].WEEK_NO.unique())
mapa_cat = (sp[["PRODUCT_ID", "COMMODITY_DESC"]].drop_duplicates()
            .set_index("PRODUCT_ID").COMMODITY_DESC)

rejilla = pd.MultiIndex.from_product(
    [vivos, semanas_test], names=["PRODUCT_ID", "WEEK_NO"]).to_frame(index=False)
rejilla["COMMODITY_DESC"] = rejilla.PRODUCT_ID.map(mapa_cat)
rejilla = rejilla.merge(sp[["PRODUCT_ID", "WEEK_NO", "unidades"]],
                        on=["PRODUCT_ID", "WEEK_NO"], how="left")
rejilla["unidades"] = rejilla.unidades.fillna(0.0)

print(f"Filas de evaluacion       : {len(rejilla):,} "
      f"({len(vivos):,} productos x {len(semanas_test)} semanas)")
print(f"Proporcion de ceros       : {(rejilla.unidades == 0).mean()*100:.1f}%")
print(f"Demanda total del periodo : {rejilla.unidades.sum():,.0f} unidades")

# ------------------------------------------- 3. pesos historicos por producto
sec("3. PESOS HISTORICOS (estimados solo con entrenamiento)")

def calcular_pesos(desde):
    """Pesos de cada producto dentro de su categoria, con historia desde 'desde'."""
    h = train_p[train_p.WEEK_NO >= desde]
    p = (h.groupby(["COMMODITY_DESC", "PRODUCT_ID"], observed=True)
         .unidades.sum().reset_index(name="uds"))
    p["peso"] = p.uds / p.groupby("COMMODITY_DESC", observed=True).uds.transform("sum")
    return p[["COMMODITY_DESC", "PRODUCT_ID", "peso"]]


# historia completa frente a ventana reciente: los pesos estaticos envejecen mal
peso = calcular_pesos(16).rename(columns={"peso": "peso_total"})
peso_rec = calcular_pesos(CORTE - 13).rename(columns={"peso": "peso_reciente"})
peso = peso.merge(peso_rec, on=["COMMODITY_DESC", "PRODUCT_ID"], how="left")
peso["peso_reciente"] = peso.peso_reciente.fillna(0.0)

print(f"Pesos calculados para {len(peso):,} pares de categoria y producto.")
comprobacion = peso.groupby("COMMODITY_DESC", observed=True).peso_total.sum()
print(f"Comprobacion de que suman uno por categoria: min={comprobacion.min():.4f}, "
      f"max={comprobacion.max():.4f}")

# cuanto se desplaza el reparto entre la historia completa y las ultimas semanas
desv = (peso.peso_total - peso.peso_reciente).abs()
print(f"\nDivergencia entre ambos repartos (desviacion absoluta media): {desv.mean():.5f}")
print(f"Productos con peso historico positivo pero nulo en las ultimas 13 semanas: "
      f"{((peso.peso_total > 0) & (peso.peso_reciente == 0)).sum():,} "
      f"({((peso.peso_total > 0) & (peso.peso_reciente == 0)).mean()*100:.1f}%)")
print("Esa cifra mide el envejecimiento del reparto: referencias que pesan en el")
print("historico pero ya no venden, y a las que un reparto estatico seguiria asignando stock.")

# ----------------------------------------------- 4. prediccion descendente
sec("4. PREDICCION DESCENDENTE")

pred_cat = pd.read_parquet(RES / "predicciones_test.parquet")[
    ["COMMODITY_DESC", "WEEK_NO", "pred", "unidades"]].rename(
    columns={"pred": "pred_categoria", "unidades": "real_categoria"})

ev = rejilla.merge(pred_cat, on=["COMMODITY_DESC", "WEEK_NO"], how="inner")
ev = ev.merge(peso, on=["COMMODITY_DESC", "PRODUCT_ID"], how="left")
ev[["peso_total", "peso_reciente"]] = ev[["peso_total", "peso_reciente"]].fillna(0.0)
ev["descendente"] = ev.pred_categoria * ev.peso_total
ev["descendente_rec"] = ev.pred_categoria * ev.peso_reciente

print(f"Filas evaluadas: {len(ev):,}")

# ----------------------------------------------- 5. predicciones ascendentes
sec("5. PREDICCIONES ASCENDENTES (referencia)")

sp_ord = sp.sort_values(["PRODUCT_ID", "WEEK_NO"])
# rejilla completa tambien en entrenamiento, para que los retardos cuenten los ceros
rej_full = pd.MultiIndex.from_product(
    [vivos, sorted(sp.WEEK_NO.unique())], names=["PRODUCT_ID", "WEEK_NO"]).to_frame(index=False)
rej_full = rej_full.merge(sp[["PRODUCT_ID", "WEEK_NO", "unidades"]],
                          on=["PRODUCT_ID", "WEEK_NO"], how="left")
rej_full["unidades"] = rej_full.unidades.fillna(0.0)
rej_full = rej_full.sort_values(["PRODUCT_ID", "WEEK_NO"])

g = rej_full.groupby("PRODUCT_ID", observed=True).unidades
rej_full["asc_ultimo"] = g.shift(1)
rej_full["asc_media4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
rej_full["asc_media_hist"] = g.shift(1).expanding().mean().reset_index(level=0, drop=True)

ev = ev.merge(rej_full[["PRODUCT_ID", "WEEK_NO", "asc_ultimo", "asc_media4", "asc_media_hist"]],
              on=["PRODUCT_ID", "WEEK_NO"], how="left")
ev = ev.dropna(subset=["asc_ultimo", "asc_media4", "asc_media_hist"])
print(f"Filas con todas las referencias disponibles: {len(ev):,}")

# ---------------------------------------------------------- 6. comparativa
sec("6. COMPARATIVA A NIVEL DE PRODUCTO")

y = ev.unidades.values
modelos = {
    "Descendente (peso histórico completo)": ev.descendente.values,
    "Descendente (peso de 13 semanas)": ev.descendente_rec.values,
    "Ascendente: media histórica": ev.asc_media_hist.values,
    "Ascendente: media móvil de 4": ev.asc_media4.values,
    "Ascendente: última semana": ev.asc_ultimo.values,
}

filas = []
for nombre, p in modelos.items():
    p = np.clip(p, 0, None)
    filas.append({
        "Modelo": nombre,
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(np.mean((y - p) ** 2)),
        "Sesgo": np.mean(p - y),
    })

tab = pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)
peor = tab.MAE.max()
tab["Mejora sobre la peor (%)"] = 100 * (peor - tab.MAE) / peor
print(tab.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
tab.to_csv(RES / "comparativa_topdown.csv", index=False)

mejor = tab.iloc[0]
ref_ultimo = tab[tab.Modelo == "Ascendente: última semana"].iloc[0]
print(f"\nMejor enfoque: {mejor.Modelo}")
print(f"Reduccion del MAE frente a la prediccion ascendente ingenua: "
      f"{100*(ref_ultimo.MAE - mejor.MAE)/ref_ultimo.MAE:.1f}%")

# ------------------------------------- 7. desglose por rotacion del producto
sec("7. DESGLOSE SEGUN LA ROTACION DEL PRODUCTO")
print("La ventaja del reparto descendente deberia crecer al bajar la rotacion.\n")

sem_train = train_p.groupby("PRODUCT_ID").WEEK_NO.nunique()
n_train = CORTE - 16
ev["rotacion"] = ev.PRODUCT_ID.map(sem_train / n_train)
ev["grupo"] = pd.cut(ev.rotacion, [-0.01, .1, .3, .6, 1.01],
                     labels=["Muy baja (<10%)", "Baja (10-30%)",
                             "Media (30-60%)", "Alta (>60%)"])

res_grupo = []
for grupo, s in ev.groupby("grupo", observed=True):
    if len(s) == 0:
        continue
    mae_td = mean_absolute_error(s.unidades, np.clip(s.descendente, 0, None))
    mae_td_r = mean_absolute_error(s.unidades, np.clip(s.descendente_rec, 0, None))
    mae_bu = mean_absolute_error(s.unidades, np.clip(s.asc_media4, 0, None))
    res_grupo.append({
        "Rotación": grupo, "Productos": s.PRODUCT_ID.nunique(),
        "Desc. histórico": mae_td, "Desc. reciente": mae_td_r, "Ascendente": mae_bu,
        "Ventaja del mejor desc. (%)": 100 * (mae_bu - min(mae_td, mae_td_r)) / mae_bu if mae_bu else np.nan,
    })

rg = pd.DataFrame(res_grupo)
print(rg.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
rg.to_csv(RES / "topdown_por_rotacion.csv", index=False)

# ------------------------------------------------------------- 8. figura
sec("8. FIGURA")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

orden = tab.sort_values("MAE")
colores = [VERDE if "Descendente" in m else AZUL for m in orden.Modelo]
ax[0].barh(orden.Modelo, orden.MAE, color=colores, height=.6)
for i, v in enumerate(orden.MAE):
    ax[0].text(v + orden.MAE.max() * .02, i, f"{v:.3f}".replace(".", ","),
               va="center", fontsize=8.5)
ax[0].invert_yaxis()
ax[0].set_xlim(0, orden.MAE.max() * 1.2)
ax[0].set_xlabel("Error absoluto medio por producto y semana")

x = np.arange(len(rg))
ax[1].bar(x - .25, rg["Desc. reciente"], .25, color=VERDE, label="Descendente (13 sem.)")
ax[1].bar(x, rg["Desc. histórico"], .25, color=AZUL, label="Descendente (completo)")
ax[1].bar(x + .25, rg["Ascendente"], .25, color=NARANJA, label="Ascendente")
ax[1].set_xticks(x)
ax[1].set_xticklabels(rg["Rotación"], fontsize=8, rotation=15, ha="right")
ax[1].set_ylabel("Error absoluto medio")
ax[1].legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig12_reparto_topdown.png")
plt.close(fig)

sec("RESUMEN")
print(f"Productos evaluados : {ev.PRODUCT_ID.nunique():,}")
print(f"Figura              : fig12_reparto_topdown.png")
print(f"Resultados          : results/comparativa_topdown.csv, topdown_por_rotacion.csv")
