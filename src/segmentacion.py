# -*- coding: utf-8 -*-
"""
Segmentacion de clientes mediante RFM y k-medias.

No es un analisis decorativo: el objetivo es responder a una pregunta concreta del
trabajo, que es si la demanda de las categorias con mayor riesgo de merma (las mas
volatiles) descansa en unos pocos hogares o esta repartida. De ahi se derivan
recomendaciones de reposicion distintas.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RAW, PROC, FIG, RES = Path("data/raw"), Path("data/processed"), Path("figures"), Path("results")
RES.mkdir(exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
PALETA = ["#2A6F97", "#E07A5F", "#4A7C59", "#8D6A9F", "#C9A227", "#C1121F"]


def sec(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


# ------------------------------------------------------------ 1. datos
sec("1. CONSTRUCCION DE LAS VARIABLES RFM")

trans = pd.read_csv(RAW / "transaction_data.csv",
                    usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID",
                             "QUANTITY", "SALES_VALUE", "WEEK_NO"])
prod = pd.read_csv(RAW / "product.csv",
                   usecols=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"])

# mismo universo que el modelado, para que las conclusiones sean comparables
ALIM = ["GROCERY", "PRODUCE", "MEAT", "MEAT-PCKGD", "DELI", "PASTRY",
        "SEAFOOD", "SEAFOOD-PCKGD", "FROZEN GROCERY", "NUTRITION"]
df = trans.merge(prod, on="PRODUCT_ID", how="left")
df = df[(df.DEPARTMENT.isin(ALIM)) & (df.COMMODITY_DESC != "COUPON/MISC ITEMS") &
        (df.WEEK_NO >= 16) & (df.QUANTITY > 0) & (df.QUANTITY <= 1000) &
        (df.SALES_VALUE > 0)].copy()

ultimo_dia = df.DAY.max()
rfm = (df.groupby("household_key")
         .agg(ultima_compra=("DAY", "max"),
              frecuencia=("BASKET_ID", "nunique"),
              importe=("SALES_VALUE", "sum"))
         .reset_index())
rfm["recencia"] = ultimo_dia - rfm.ultima_compra

print(f"Hogares analizados: {len(rfm):,}")
print(f"Periodo: dias {df.DAY.min()} a {ultimo_dia}\n")
print(rfm[["recencia", "frecuencia", "importe"]].describe()
      .to_string(float_format=lambda x: f"{x:,.1f}"))

# las tres variables estan muy sesgadas a la derecha: se transforman antes de agrupar
X = np.column_stack([
    np.log1p(rfm.recencia), np.log1p(rfm.frecuencia), np.log1p(rfm.importe)
])
Xs = StandardScaler().fit_transform(X)

# --------------------------------------------- 2. eleccion del numero de grupos
sec("2. ELECCION DEL NUMERO DE GRUPOS")
inercias, siluetas, ks = [], [], range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xs)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(Xs, km.labels_))
    print(f"  k={k}:  inercia={km.inertia_:9,.0f}   silueta={siluetas[-1]:.3f}")

K = int(list(ks)[int(np.argmax(siluetas))])
K_FINAL = 4 if K < 4 else K
print(f"\nMaximo coeficiente de silueta en k={K} ({max(siluetas):.3f}).")
print(f"Numero de grupos adoptado: {K_FINAL} (silueta {siluetas[list(ks).index(K_FINAL)]:.3f}).")
print("Justificacion de la discrepancia: con k=2 la particion es mas compacta pero solo")
print("distingue entre hogares activos e inactivos, lo que carece de valor operativo. Con")
print("k=4 se pierde cohesion pero aparecen grados de intensidad de compra accionables.")
print("La eleccion prioriza la interpretabilidad sobre la metrica interna y debe declararse")
print("expresamente en la memoria, porque es una decision del analista.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(list(ks), inercias, "-o", color=PALETA[0], ms=5)
ax[0].set_xlabel("Número de grupos"); ax[0].set_ylabel("Inercia")
ax[1].plot(list(ks), siluetas, "-o", color=PALETA[1], ms=5)
ax[1].axvline(K_FINAL, color="#C1121F", ls="--", lw=1.3)
ax[1].set_xlabel("Número de grupos"); ax[1].set_ylabel("Coeficiente de silueta")
fig.tight_layout()
fig.savefig(FIG / "fig10_eleccion_k.png")
plt.close(fig)

# --------------------------------------------------------- 3. segmentacion
sec("3. CARACTERIZACION DE LOS SEGMENTOS")
km = KMeans(n_clusters=K_FINAL, n_init=30, random_state=42).fit(Xs)
rfm["segmento"] = km.labels_

perfil = (rfm.groupby("segmento")
            .agg(hogares=("household_key", "size"),
                 recencia_mediana=("recencia", "median"),
                 frecuencia_mediana=("frecuencia", "median"),
                 importe_mediano=("importe", "median"),
                 importe_total=("importe", "sum")))
perfil["% hogares"] = 100 * perfil.hogares / perfil.hogares.sum()
perfil["% facturacion"] = 100 * perfil.importe_total / perfil.importe_total.sum()
print(perfil.to_string(float_format=lambda x: f"{x:,.1f}"))

# etiquetas por orden de intensidad de compra, sin ambiguedad ni repeticiones
orden = perfil.sort_values("importe_mediano", ascending=False).index.tolist()
etiquetas = ["Intensivos", "Habituales", "Ocasionales", "Inactivos",
             "Residuales", "Marginales"]
nombres = {seg: etiquetas[i] for i, seg in enumerate(orden)}

rfm["nombre"] = rfm.segmento.map(nombres)
print("\nEtiquetas asignadas:")
for seg in orden:
    p = perfil.loc[seg]
    print(f"  {nombres[seg]:<14} {p['% hogares']:5.1f}% de hogares  "
          f"{p['% facturacion']:5.1f}% de facturacion  "
          f"(frecuencia mediana {p.frecuencia_mediana:.0f} cestas)")

# ------------------------------- 4. conexion con el objetivo del trabajo
sec("4. QUIEN SOSTIENE LA DEMANDA DE LAS CATEGORIAS DE RIESGO")
print("Las categorias mas volatiles son las de mayor riesgo de merma. Interesa saber")
print("si su demanda descansa en pocos hogares o esta repartida.\n")

vol = pd.read_csv(PROC / "volatilidad_categorias.csv", index_col=0).dropna(subset=["cv"])
volatiles = vol.sort_values("cv", ascending=False).head(20).index.tolist()
estables = vol.sort_values("cv").head(20).index.tolist()

d = df.merge(rfm[["household_key", "nombre"]], on="household_key", how="left")

# Pregunta correcta: que proporcion DE SU PROPIO gasto dedica cada segmento a las
# categorias de riesgo. Comparar cuotas absolutas solo reflejaria el tamanio del grupo.
gasto_seg = d.groupby("nombre").SALES_VALUE.sum()
comp = pd.DataFrame({
    "Cuota de la facturación (%)": 100 * gasto_seg / gasto_seg.sum(),
    "Su gasto en volátiles (%)": 100 * d[d.COMMODITY_DESC.isin(volatiles)]
                                  .groupby("nombre").SALES_VALUE.sum() / gasto_seg,
    "Su gasto en estables (%)": 100 * d[d.COMMODITY_DESC.isin(estables)]
                                 .groupby("nombre").SALES_VALUE.sum() / gasto_seg,
}).sort_values("Cuota de la facturación (%)", ascending=False)
print(comp.to_string(float_format=lambda x: f"{x:,.2f}"))

rango = comp["Su gasto en volátiles (%)"].max() - comp["Su gasto en volátiles (%)"].min()
print(f"\nRecorrido entre segmentos en el peso de las categorias volatiles: "
      f"{rango:.2f} puntos porcentuales.")
if rango < 2:
    print("Interpretacion: la diferencia es minima. La composicion de clientes NO explica")
    print("que una categoria sea volatil, luego la volatilidad es intrinseca a la categoria.")
    print("Es un resultado negativo, y respalda la decision de modelar por categoria sin")
    print("incorporar la segmentacion como variable predictora.")
else:
    print("Interpretacion: existen diferencias apreciables entre segmentos.")
comp.to_csv(RES / "segmentos_vs_categorias.csv")

conc = d.groupby("household_key").SALES_VALUE.sum().sort_values(ascending=False)
top20 = 100 * conc.head(int(len(conc) * .2)).sum() / conc.sum()
print(f"\nConcentracion: el 20% de los hogares genera el {top20:.1f}% de la facturacion.")

# ------------------------------------------------------------ 5. figuras
sec("5. FIGURAS")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, seg in enumerate(orden):
    s = rfm[rfm.segmento == seg]
    ax[0].scatter(s.frecuencia, s.importe, s=13, alpha=.55,
                  color=PALETA[i % len(PALETA)], label=nombres[seg])
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("Número de cestas de compra"); ax[0].set_ylabel("Gasto acumulado")
ax[0].legend(fontsize=8.5)

anchura = 0.38
pos = np.arange(len(comp))
ax[1].barh(pos + anchura / 2, comp["Su gasto en estables (%)"], anchura,
           color=PALETA[0], label="En categorías estables")
ax[1].barh(pos - anchura / 2, comp["Su gasto en volátiles (%)"], anchura,
           color=PALETA[1], label="En categorías volátiles")
ax[1].set_yticks(pos); ax[1].set_yticklabels(comp.index)
ax[1].set_xlabel("Porcentaje del gasto del propio segmento")
ax[1].legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig11_segmentos.png")
plt.close(fig)

# --------------------------------------------- 6. cruce con demografia
sec("6. CRUCE CON LA DEMOGRAFIA DISPONIBLE")
demo = pd.read_csv(RAW / "hh_demographic.csv")
cruce = rfm.merge(demo, on="household_key", how="inner")
print(f"Hogares con demografia conocida: {len(cruce):,} de {len(rfm):,} "
      f"({100*len(cruce)/len(rfm):.1f}%). La cobertura parcial limita este analisis.\n")

tabla_demo = pd.crosstab(cruce.nombre, cruce.HOUSEHOLD_SIZE_DESC, normalize="index") * 100
print("Distribucion del tamanio del hogar por segmento (%):")
print(tabla_demo.to_string(float_format=lambda x: f"{x:,.1f}"))
tabla_demo.to_csv(RES / "segmentos_vs_demografia.csv")

rfm.to_csv(RES / "segmentos_hogares.csv", index=False)
perfil.to_csv(RES / "perfil_segmentos.csv")

sec("RESUMEN")
print(f"Segmentos identificados : {K_FINAL}")
print(f"Figuras                 : fig10, fig11")
print(f"Resultados              : results/segmentos_*.csv")
