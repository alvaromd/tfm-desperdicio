# -*- coding: utf-8 -*-
"""
Genera las figuras definitivas para la memoria a partir de los datos ya procesados.
Criterios editoriales: sin titulo incrustado (el titulo va en el pie de figura de
Word), etiquetas en espanol con tildes, tamano y resolucion aptos para impresion.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RAW, FIG, PROC = Path("data/raw"), Path("figures"), Path("data/processed")
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "legend.frameon": False,
})

AZUL, NARANJA, VERDE, MORADO, ROJO = "#2A6F97", "#E07A5F", "#4A7C59", "#8D6A9F", "#C1121F"

serie = pd.read_parquet(PROC / "serie_semanal_global.parquet")
panel_mod = pd.read_parquet(PROC / "panel_modelado.parquet")
vol = pd.read_csv(PROC / "volatilidad_categorias.csv", index_col=0)

trans = pd.read_csv(RAW / "transaction_data.csv",
                    usecols=["household_key", "PRODUCT_ID", "QUANTITY", "SALES_VALUE", "WEEK_NO"])

# --- figura 1
fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
ax[0].plot(serie.WEEK_NO, serie.ventas / 1000, lw=1.5, color=AZUL)
ax[0].set_ylabel("Facturación semanal\n(miles de unidades monetarias)")
ax[1].plot(serie.WEEK_NO, serie.hogares, lw=1.5, color=NARANJA)
ax[1].set_ylabel("Hogares activos")
ax[1].set_xlabel("Semana")
for a in ax:
    a.axvspan(1, 15, color="#C1121F", alpha=0.07)
ax[0].annotate("Periodo descartado", xy=(8, ax[0].get_ylim()[1] * 0.55),
               ha="center", fontsize=8.5, color=ROJO)
fig.tight_layout()
fig.savefig(FIG / "fig01_evolucion_semanal.png")
plt.close(fig)

# --- figura 2
h = trans.groupby("WEEK_NO").household_key.nunique()
meseta = h[h.index >= 30].median()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h.index, h.values, lw=1.6, color=AZUL)
ax.axhline(meseta, color="gray", ls=":", lw=1.2)
ax.axvline(16, color=ROJO, ls="--", lw=1.4)
ax.annotate(f"Meseta: {meseta:,.0f} hogares".replace(",", "."),
            xy=(60, meseta), xytext=(60, meseta * 0.72), fontsize=9, color="gray")
ax.annotate("Semana 16:\ninicio del periodo válido", xy=(16, meseta * 0.5),
            xytext=(24, meseta * 0.32), fontsize=9, color=ROJO,
            arrowprops=dict(arrowstyle="->", color=ROJO, lw=1))
ax.set_xlabel("Semana")
ax.set_ylabel("Hogares con al menos una compra")
fig.tight_layout()
fig.savefig(FIG / "fig02_rampa_hogares.png")
plt.close(fig)

# --- figura 3
rot = trans.groupby("PRODUCT_ID").WEEK_NO.nunique()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(rot, bins=51, color=MORADO, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Número de semanas con venta (sobre un total de 102)")
ax.set_ylabel("Número de productos")
pct1 = (rot == 1).mean() * 100
pct10 = (rot < 10.2).mean() * 100
ax.annotate(f"El {pct1:.0f} % de los productos\nvende en una sola semana",
            xy=(1, (rot == 1).sum()), xytext=(22, (rot == 1).sum() * 0.82),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=1, color="#404040"))
fig.tight_layout()
fig.savefig(FIG / "fig03_rotacion_productos.png")
plt.close(fig)

# --- figura 4
v = vol.dropna(subset=["cv"])
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(v.cv, bins=40, color=VERDE, edgecolor="white", linewidth=0.4)
ax.axvline(v.cv.median(), color=ROJO, ls="--", lw=1.4,
           label=f"Mediana = {v.cv.median():.2f}".replace(".", ","))
ax.set_xlabel("Coeficiente de variación de la demanda semanal")
ax.set_ylabel("Número de categorías")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig04_volatilidad_categorias.png")
plt.close(fig)

# --- figura 5
top6 = (panel_mod.groupby("COMMODITY_DESC").unidades.sum()
        .sort_values(ascending=False).head(6).index)
nombres = {
    "SOFT DRINKS": "Refrescos", "BEEF": "Vacuno", "FLUID MILK PRODUCTS": "Leche",
    "CHEESE": "Queso", "BAG SNACKS": "Aperitivos", "BAKED BREAD/BUNS/ROLLS": "Pan",
    "FROZEN PIZZA": "Pizza congelada", "COLD CEREAL": "Cereales",
    "FRZN MEAT/MEAT DINNERS": "Platos congelados", "DELI MEATS": "Embutidos",
    "POTATOES": "Patatas", "BANANAS": "Plátanos", "EGGS": "Huevos",
    "LUNCHMEAT": "Fiambre", "CHICKEN": "Pollo", "SOUP": "Sopa",
    "YOGURT": "Yogur", "ICE CREAM/MILK/SHERBTS": "Helados",
    "CARBONATED SOFT DRINKS": "Refrescos con gas", "SALD DRSNG/SNDWCH SPRD": "Salsas",
    "TROPICAL FRUIT": "Fruta tropical", "VEGETABLES SALAD": "Ensaladas",
    "BEERS/ALES": "Cervezas", "CANDY - PACKAGED": "Golosinas",
}
fig, ax = plt.subplots(figsize=(9, 4.6))
colores = [AZUL, NARANJA, VERDE, ROJO, MORADO, "#8C6D46"]
for c, col in zip(top6, colores):
    s = panel_mod[panel_mod.COMMODITY_DESC == c].sort_values("WEEK_NO")
    # serie original tenue y media movil de 4 semanas destacada
    ax.plot(s.WEEK_NO, s.unidades_por_hogar, lw=0.7, color=col, alpha=0.28)
    ax.plot(s.WEEK_NO, s.unidades_por_hogar.rolling(4, center=True).mean(),
            lw=2.0, color=col, label=nombres.get(c, c.title()))
ax.set_xlabel("Semana")
ax.set_ylabel("Unidades por hogar activo")
ax.legend(ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout()
fig.savefig(FIG / "fig05_demanda_normalizada.png")
plt.close(fig)

print("Figuras definitivas generadas:")
for f in sorted(FIG.glob("fig0*.png")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
