# -*- coding: utf-8 -*-
"""
Figuras conceptuales para la memoria, de elaboracion propia.

A diferencia de las figuras de resultados, estas explican conceptos y no datos:
  fig13  Flujo metodologico completo del trabajo
  fig14  El problema del vendedor de periodicos y el cuantil optimo
  fig15  La cadena alimentaria y los dos indicadores que la miden
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from scipy import stats

FIG = Path("figures")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9.5,
})
AZUL, NARANJA, VERDE, ROJO, GRIS = "#2A6F97", "#E07A5F", "#4A7C59", "#C1121F", "#6C757D"


def caja(ax, x, y, w, h, texto, color, fc=None, fs=9, negrita=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.4, edgecolor=color,
                                facecolor=fc if fc else color + "18"))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
            color="#1a1a1a", weight="bold" if negrita else "normal", linespacing=1.45)


def flecha(ax, p1, p2, color=GRIS, estilo="-|>", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=estilo, mutation_scale=13,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# ============================================ fig13: flujo metodologico
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")

caja(ax, .3, 5.5, 2.5, .75, "Datos de transacciones\n2.595.732 líneas", AZUL, fs=9)
caja(ax, 3.3, 5.5, 2.5, .75, "Catálogo de productos\n92.353 referencias", AZUL, fs=9)
caja(ax, 6.3, 5.5, 3.3, .75, "Demografía y promociones\n(no utilizadas en el modelo)", GRIS, fs=8.7)

caja(ax, 1.4, 4.25, 6.2, .8,
     "Filtrado: departamentos de alimentación, exclusión de artefactos,\n"
     "periodo válido desde la semana 16   →   80,6 % de las líneas retenidas", NARANJA, fs=9)

caja(ax, 2.6, 3.05, 3.9, .75, "Panel semana × categoría\n157 categorías × 87 semanas", VERDE, fs=9, negrita=True)

caja(ax, .25, 1.9, 2.5, .78, "Ingeniería de variables\nretardos, medias móviles,\nestacionalidad", AZUL, fs=8.4)
caja(ax, 3.1, 1.9, 2.5, .78, "Partición temporal\ncorte en la semana 88\n(nunca aleatoria)", AZUL, fs=8.4)
caja(ax, 5.95, 1.9, 2.8, .78, "Modelos comparados\nlíneas base, XGBoost, SVR\ny XGBoost cuantílico", AZUL, fs=8.4)

caja(ax, 1.0, .45, 3.3, .78, "Predicción de demanda\npor categoría y semana", VERDE, fs=9)
caja(ax, 5.3, .45, 4.0, .78, "Frontera excedente-rotura\nelección del punto de operación", ROJO, fs=9, negrita=True)

flecha(ax, (1.55, 5.5), (2.9, 5.05)); flecha(ax, (4.55, 5.5), (4.5, 5.05))
flecha(ax, (7.95, 5.5), (6.3, 5.05), color="#adb5bd")
flecha(ax, (4.5, 4.25), (4.5, 3.8))
flecha(ax, (4.4, 3.05), (1.5, 2.7))
flecha(ax, (2.75, 2.29), (3.1, 2.29))
flecha(ax, (5.6, 2.29), (5.95, 2.29))
flecha(ax, (6.6, 1.9), (3.0, 1.25))
flecha(ax, (7.7, 1.9), (7.4, 1.25))
flecha(ax, (4.3, .84), (5.3, .84), color=ROJO)

ax.text(5, 6.42, "Flujo metodológico del trabajo", ha="center", fontsize=11.5, weight="bold", color="#1a1a1a")
fig.tight_layout()
fig.savefig(FIG / "fig13_flujo_metodologico.png")
plt.close(fig)

# ================================ fig14: el vendedor de periodicos
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))

x = np.linspace(0, 200, 600)
dens = stats.norm.pdf(x, 100, 26)

for k, (alpha, titulo, color) in enumerate([
        (0.35, "Coste de exceso alto\n(producto perecedero)", VERDE),
        (0.85, "Coste de rotura alto\n(producto estratégico)", NARANJA)]):
    a = ax[k]
    q = stats.norm.ppf(alpha, 100, 26)
    a.plot(x, dens, color="#495057", lw=1.6)
    a.fill_between(x[x <= q], dens[x <= q], color=color, alpha=.30)
    a.axvline(q, color=color, lw=2.2)
    a.axvline(100, color=GRIS, lw=1.3, ls=":")
    a.text(100, dens.max() * .06, "demanda\nesperada", ha="center", va="bottom",
           fontsize=8.2, color=GRIS, linespacing=1.3)
    dx, ali = (7, "left") if alpha > .5 else (-7, "right")
    a.text(q + dx, dens.max() * .82, "cantidad\na reponer", ha=ali, fontsize=8.6,
           color=color, weight="bold", linespacing=1.3)
    a.annotate("", xy=(q, dens.max() * .52), xytext=(100, dens.max() * .52),
               arrowprops=dict(arrowstyle="<->", color=color, lw=1.3))
    a.set_ylim(0, dens.max() * 1.20)
    a.set_title(titulo, fontsize=9.6, color="#1a1a1a", pad=12)
    a.set_xlabel("Unidades de demanda")
    a.set_yticks([]); a.set_xticks([])
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.text(.02, .95, f"cuantil = {alpha:.2f}".replace(".", ","), transform=a.transAxes,
           fontsize=9, color=color, weight="bold")

fig.suptitle("La cantidad óptima a reponer es un cuantil de la demanda, no su media",
             fontsize=11, weight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "fig14_vendedor_periodicos.png")
plt.close(fig)

# ============================ fig15: cadena alimentaria e indicadores
fig, ax = plt.subplots(figsize=(10.5, 3.9))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.9); ax.axis("off")

etapas = [("Producción", .15), ("Poscosecha,\nalmacenamiento\ny transporte", 1.75),
          ("Procesado", 3.35), ("Comercio\nminorista", 5.25),
          ("Restauración", 6.95), ("Hogares", 8.65)]
for nombre, x0 in etapas:
    es_foco = nombre.startswith("Comercio")
    caja(ax, x0, 2.05, 1.5, .8, nombre,
         ROJO if es_foco else GRIS,
         fc="#C1121F22" if es_foco else "#f1f3f5",
         fs=8.4, negrita=es_foco)
    if x0 < 8.6:
        flecha(ax, (x0 + 1.5, 2.45), (x0 + 1.72, 2.45), color="#adb5bd", lw=1.2)

ax.plot([.15, 4.85], [1.62, 1.62], color=AZUL, lw=2.6)
ax.plot([5.25, 10.15], [1.62, 1.62], color=NARANJA, lw=2.6)
for xx, col in [(.15, AZUL), (4.85, AZUL), (5.25, NARANJA), (10.15, NARANJA)]:
    ax.plot([xx, xx], [1.5, 1.74], color=col, lw=2.6)

ax.text(2.5, 1.18, "PÉRDIDA de alimentos", ha="center", fontsize=9.6, weight="bold", color=AZUL)
ax.text(2.5, .82, "Índice de Pérdida de Alimentos (FAO)\n13-14 % según la estimación",
        ha="center", fontsize=8.4, color=AZUL, linespacing=1.4)
ax.text(7.7, 1.18, "DESPERDICIO de alimentos", ha="center", fontsize=9.6, weight="bold", color=NARANJA)
ax.text(7.7, .82, "Índice de Desperdicio de Alimentos (PNUMA)\n19 % de los alimentos disponibles",
        ha="center", fontsize=8.4, color=NARANJA, linespacing=1.4)

ax.text(5.25, .28, "Los dos indicadores miden tramos distintos de la cadena y no pueden sumarse",
        ha="center", fontsize=8.6, style="italic", color="#495057")
ax.text(5.25, 3.55, "Dónde se mide la pérdida y dónde el desperdicio",
        ha="center", fontsize=11.2, weight="bold", color="#1a1a1a")
ax.text(6.0, 3.15, "ámbito de este trabajo ↓", ha="center", fontsize=8.6, color=ROJO, weight="bold")

fig.tight_layout()
fig.savefig(FIG / "fig15_cadena_indicadores.png")
plt.close(fig)

print("Figuras conceptuales generadas:")
for f in sorted(FIG.glob("fig1[345]*.png")):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
