# -*- coding: utf-8 -*-
"""
Figuras de la comparativa ampliada con los modelos de redes neuronales.

  fig06  comparativa de error de los trece modelos, agrupados por familia
  fig18  los trece modelos situados en el plano excedente-rotura, con la
         frontera de XGBoost como referencia

La segunda es la que evita leer mal la tabla de impacto: un modelo que
subestima de forma sistematica reduce el excedente a costa de la rotura,
y eso solo se ve situandolo en el plano.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RES, FIG = Path("results"), Path("figures")
AZUL, NARANJA, VERDE, ROJO, GRIS = "#2A6F97", "#E07A5F", "#4A7C59", "#C1121F", "#6C757D"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.grid": True, "grid.alpha": 0.25})

# La tabla de los trece modelos se monta a partir de las salidas de modelado.py,
# red_neuronal.py y redes_avanzadas.py, y se guarda para consulta.
NOMBRE = {
    "XGBoost cuantilico (a=0.45)": "XGBoost cuantílico (α=0,45)",
    "XGBoost cuantilico (a=0.35)": "XGBoost cuantílico (α=0,35)",
    "Media movil de 4 semanas": "Media móvil de 4 semanas",
    "SVR (nucleo RBF)": "SVR (núcleo RBF)",
    "Ingenuo (semana anterior)": "Ingenuo",
    "Red neuronal (perdida simetrica)": "MLP (pérdida simétrica)",
    "Red neuronal cuantilica (a=0.45)": "MLP cuantílico (α=0,45)",
    "Red neuronal cuantilica (a=0.35)": "MLP cuantílico (α=0,35)",
    "Red multicuantil (salida a=0,35)": "Red multicuantil (α=0,35)",
}
COLS = ["Modelo", "MAE", "Excedente (uds)", "Rotura (uds)",
        "Excedente sobre demanda (%)", "Reduccion del excedente (%)"]
xgb = pd.read_csv(RES / "comparativa_modelos.csv").merge(
    pd.read_csv(RES / "impacto_excedente.csv"), on="Modelo")
t = pd.concat([xgb, pd.read_csv(RES / "red_neuronal.csv"),
               pd.read_csv(RES / "redes_avanzadas.csv")], ignore_index=True)
t["Modelo"] = t.Modelo.map(lambda m: NOMBRE.get(m, m))
t = t[COLS].sort_values("Excedente (uds)").reset_index(drop=True)
t.to_csv(RES / "comparativa_completa.csv", index=False)

CORTO = {
    "XGBoost cuantílico (α=0,45)": "XGBoost cuant. α=0,45",
    "XGBoost cuantílico (α=0,35)": "XGBoost cuant. α=0,35",
    "MLP cuantílico (α=0,45)": "MLP cuant. α=0,45",
    "MLP cuantílico (α=0,35)": "MLP cuant. α=0,35",
    "LSTM cuantílico (α=0,35)": "LSTM cuant. α=0,35",
    "Red con embeddings (α=0,35)": "Embeddings α=0,35",
    "Red multicuantil (α=0,35)": "Multicuantil α=0,35",
    "MLP (pérdida simétrica)": "MLP simétrico",
    "LSTM (pérdida simétrica)": "LSTM simétrico",
    "Media móvil de 4 semanas": "Media móvil de 4",
    "SVR (núcleo RBF)": "SVR",
}
t["corto"] = t.Modelo.map(lambda m: CORTO.get(m, m))

BASE = {"Ingenuo", "Media móvil de 4 semanas"}


def familia(m):
    if m in BASE:
        return "base"
    return "cuantil" if ("cuant" in m or "embeddings" in m or "multicuantil" in m) else "simetrico"


t["familia"] = t.Modelo.map(familia)
COLOR = {"base": GRIS, "simetrico": NARANJA, "cuantil": VERDE}
ETIQ = {"base": "Líneas base", "simetrico": "Pérdida simétrica", "cuantil": "Pérdida cuantílica"}

# --- fig06: comparativa de error
o = t.sort_values("MAE")
fig, ax = plt.subplots(figsize=(8.4, 5.4))
ax.barh(o.corto, o.MAE, color=[COLOR[f] for f in o.familia], height=0.66)
for i, v in enumerate(o.MAE):
    ax.text(v + o.MAE.max() * 0.012, i, f"{v:,.2f}".replace(".", ","), va="center", fontsize=8.6)
ax.set_xlabel("Error absoluto medio (unidades)")
# margen extra a la derecha: la leyenda va arriba, donde las barras son cortas
ax.set_xlim(0, o.MAE.max() * 1.34)
ax.invert_yaxis()
man = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in ["base", "simetrico", "cuantil"]]
ax.legend(man, [ETIQ[k] for k in ["base", "simetrico", "cuantil"]],
          loc="upper right", fontsize=8.6, framealpha=0.95)
fig.tight_layout()
fig.savefig(FIG / "fig06_comparativa_modelos.png")
plt.close(fig)
print("fig06 regenerada con los trece modelos")

# --- fig18: plano excedente-rotura
fr = pd.read_csv(RES / "frontera_excedente_rotura.csv")
t["rot_pct"] = 100 * t["Rotura (uds)"] / t["Rotura (uds)"].sum() * 0  # placeholder
DEM = t.loc[t.Modelo == "Ingenuo", "Excedente (uds)"].iloc[0] / (
    t.loc[t.Modelo == "Ingenuo", "Excedente sobre demanda (%)"].iloc[0] / 100)
t["rot_pct"] = 100 * t["Rotura (uds)"] / DEM
t["exc_pct"] = t["Excedente sobre demanda (%)"]

fig, ax = plt.subplots(figsize=(8.6, 5.6))
ax.plot(fr.exc_pct, fr.rot_pct, "-", color=AZUL, lw=1.6, alpha=.75, zorder=1,
        label="Frontera del XGBoost cuantílico")
MARCA = {"base": "s", "simetrico": "^", "cuantil": "o"}
for f in ["base", "simetrico", "cuantil"]:
    s = t[t.familia == f]
    ax.scatter(s.exc_pct, s.rot_pct, s=68, marker=MARCA[f], color=COLOR[f],
               edgecolors="white", linewidths=.7, zorder=3, label=ETIQ[f])
# colocacion manual de las etiquetas: con trece puntos muy juntos, el
# desplazamiento automatico las solapa
POS = {
    "LSTM cuant. α=0,35": (0, 11, "center"), "MLP cuant. α=0,35": (9, 3, "left"),
    "XGBoost cuant. α=0,35": (-9, -4, "right"), "Embeddings α=0,35": (0, -15, "center"),
    "Multicuantil α=0,35": (9, 4, "left"), "LSTM simétrico": (9, -9, "left"),
    "XGBoost cuant. α=0,45": (-9, 4, "right"), "MLP cuant. α=0,45": (9, 5, "left"),
    "Media móvil de 4": (-9, 3, "right"), "MLP simétrico": (9, 0, "left"),
    "XGBoost": (0, -15, "center"), "SVR": (0, 10, "center"), "Ingenuo": (9, 3, "left"),
}
for _, r in t.iterrows():
    dx, dy, ha = POS.get(r.corto, (8, 5, "left"))
    ax.annotate(r.corto, (r.exc_pct, r.rot_pct), textcoords="offset points",
                xytext=(dx, dy), fontsize=7.6, color="#33393f", ha=ha)
ax.set_xlabel("Excedente de reposición sobre la demanda (%)")
ax.set_ylabel("Rotura de stock sobre la demanda (%)")
ax.legend(loc="upper right", fontsize=8.6)
fig.tight_layout()
fig.savefig(FIG / "fig18_plano_excedente_rotura.png")
plt.close(fig)
print("fig18 generada: los trece modelos en el plano excedente-rotura")

print("\nModelos por debajo de la frontera de XGBoost (no dominados):")
for _, r in t.iterrows():
    interp = np.interp(r.exc_pct, fr.exc_pct, fr.rot_pct)
    if r.rot_pct <= interp + 0.05:
        print(f"   {r.corto:<24} exc {r.exc_pct:5.1f} %   rot {r.rot_pct:5.1f} %")
