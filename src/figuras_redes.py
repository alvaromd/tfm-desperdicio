# -*- coding: utf-8 -*-
"""
Diagramas de las cuatro arquitecturas de red del trabajo.

Las dimensiones se toman del codigo que las entrena, red_neuronal.py y
redes_avanzadas.py, y no de la memoria. Las tres variantes comparten el
tronco de 64 y 32 unidades del perceptron base: lo que cambia es la entrada,
en la recurrente y la de embeddings, o la salida, en la multicuantil.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG = Path("figures")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9.5,
})
AZUL, NARANJA, VERDE, ROJO, GRIS = "#2A6F97", "#E07A5F", "#4A7C59", "#C1121F", "#6C757D"


def caja(ax, x, y, w, h, texto, color, fs=8.6, negrita=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, edgecolor=color, facecolor=color + "18"))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
            color="#1a1a1a", weight="bold" if negrita else "normal", linespacing=1.4)


def flecha(ax, p1, p2, color=GRIS, lw=1.3):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color))


def pila(ax, x, ancho, capas, y0=0.55, alto=0.62, hueco=0.42):
    """Dibuja una columna de capas de abajo arriba y devuelve la altura final."""
    y = y0
    centros = []
    for texto, color, negrita in capas:
        caja(ax, x, y, ancho, alto, texto, color, negrita=negrita)
        centros.append(y + alto / 2)
        y += alto + hueco
    for a, b in zip(centros[:-1], centros[1:]):
        flecha(ax, (x + ancho / 2, a + alto / 2), (x + ancho / 2, b - alto / 2))
    return y


# Dos filas de dos: a lo ancho de una pagina vertical, cuatro paneles en
# fila dejarian el texto demasiado pequeno para leerse en papel.
fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.4))
TRONCO = [("64 unidades\nReLU + abandono 10 %", AZUL, False),
          ("32 unidades\nReLU", AZUL, False)]

paneles = [
    ("a) Perceptrón multicapa",
     [("14 variables\nconstruidas", VERDE, False)] + TRONCO + [("1 salida\npérdida cuantílica", ROJO, True)]),
    ("b) Red multicuantil",
     [("14 variables\nconstruidas", VERDE, False)] + TRONCO +
     [("5 salidas\nα = 0,20 a 0,80", ROJO, True)]),
    ("c) Red recurrente",
     [("13 semanas × 2 series\ndemanda y hogares", VERDE, False),
      ("LSTM\n32 unidades ocultas", NARANJA, False),
      ("último paso\nde la secuencia", NARANJA, False),
      ("1 salida\npérdida cuantílica", ROJO, True)]),
    ("d) Red con embeddings",
     [("14 variables  +  categoría\n157 categorías → 8 dimensiones", VERDE, False)] + TRONCO +
     [("1 salida\npérdida cuantílica", ROJO, True)]),
]

for ax, (titulo, capas) in zip(axes.ravel(), paneles):
    ax.set_xlim(0, 3.4)
    ax.set_ylim(0.35, 4.6)
    ax.axis("off")
    pila(ax, 0.2, 3.0, capas)
    ax.set_title(titulo, fontsize=10, weight="bold", color="#1a1a1a", pad=6)

fig.tight_layout()
destino = FIG / "fig_redes_arquitecturas.png"
fig.savefig(destino)
plt.close(fig)
print(f"Guardada {destino}  ({destino.stat().st_size / 1024:.1f} KB)")
