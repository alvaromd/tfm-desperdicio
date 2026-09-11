# -*- coding: utf-8 -*-
"""
Tres experimentos con redes neuronales sobre el panel semana x categoria.

  A. Red multicuantil : una sola red con cinco salidas, una por cuantil.
                        Traza la frontera completa sin reentrenar.
  B. Red recurrente   : LSTM sobre la secuencia cruda de demanda, sin las
                        variables construidas a mano del apartado 4.4.
  C. Embeddings       : perceptron con un vector aprendido por categoria,
                        capacidad que XGBoost no tiene de forma nativa.

Todos comparten panel, particion temporal y semilla con red_neuronal.py.
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEMILLA, CORTE, CORTE_VAL, COLCHON = 42, 88, 80, 0.10
VENTANA = 13                       # semanas de historia para el LSTM
CUANTILES = [0.20, 0.35, 0.50, 0.65, 0.80]
PROC, RES, FIG = Path("data/processed"), Path("results"), Path("figures")
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
AZUL, NARANJA, VERDE, ROJO, GRIS = "#2A6F97", "#E07A5F", "#4A7C59", "#C1121F", "#6C757D"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "font.size": 9.5,
                     "axes.grid": True, "grid.alpha": 0.25})


def sec(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# ============================================================ datos comunes
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

NIVEL = [f for f in FEATURES if f.startswith(("lag_", "media_", "desv_", "nivel_", "hogares_"))]
idx_nivel = [FEATURES.index(f) for f in NIVEL]


def a_log(X):
    Z = X.copy(); Z[:, idx_nivel] = np.log1p(np.clip(Z[:, idx_nivel], 0, None)); return Z


tr = datos[datos.WEEK_NO < CORTE_VAL]
va = datos[(datos.WEEK_NO >= CORTE_VAL) & (datos.WEEK_NO < CORTE)]
te = datos[datos.WEEK_NO >= CORTE]
esc = StandardScaler().fit(a_log(tr[FEATURES].values))
Xtr = torch.tensor(esc.transform(a_log(tr[FEATURES].values)), dtype=torch.float32)
Xva = torch.tensor(esc.transform(a_log(va[FEATURES].values)), dtype=torch.float32)
Xte = torch.tensor(esc.transform(a_log(te[FEATURES].values)), dtype=torch.float32)
y_tr, y_va, y_te = (d.unidades.values.astype(np.float32) for d in (tr, va, te))
SIGMA = float(y_tr.std())
Ytr = torch.tensor(y_tr / SIGMA).unsqueeze(1)
Yva = torch.tensor(y_va / SIGMA).unsqueeze(1)

MAE_INGENUO, EXC_INGENUO = 43.43524416135881, 88181.9
DEMANDA = float(y_te.sum())

cats = sorted(datos.COMMODITY_DESC.unique())
cat2id = {c: i for i, c in enumerate(cats)}
Itr = torch.tensor(tr.COMMODITY_DESC.map(cat2id).values, dtype=torch.long)
Iva = torch.tensor(va.COMMODITY_DESC.map(cat2id).values, dtype=torch.long)
Ite = torch.tensor(te.COMMODITY_DESC.map(cat2id).values, dtype=torch.long)


def metricas(nombre, pred):
    pred = np.clip(pred, 0, None)
    m = y_te > 0
    rep = pred * (1 + COLCHON)
    exc = float(np.clip(rep - y_te, 0, None).sum()); rot = float(np.clip(y_te - rep, 0, None).sum())
    return {"Modelo": nombre,
            "MAE": mean_absolute_error(y_te, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "MAPE (%)": float(np.mean(np.abs((y_te[m] - pred[m]) / y_te[m])) * 100),
            "Mejora sobre ingenuo (%)": 100 * (MAE_INGENUO - mean_absolute_error(y_te, pred)) / MAE_INGENUO,
            "Excedente (uds)": exc, "Rotura (uds)": rot,
            "Excedente sobre demanda (%)": 100 * exc / DEMANDA,
            "Reduccion del excedente (%)": 100 * (EXC_INGENUO - exc) / EXC_INGENUO}


def pinball(alphas):
    a = torch.tensor(alphas, dtype=torch.float32)
    def f(pred, real):
        e = real - pred
        return torch.mean(torch.maximum(a * e, (a - 1) * e))
    return f


def entrenar(red, perdida, datos_tr, datos_va, etiqueta, epocas=400, paciencia=40, lr=1e-3, lote=256):
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    n = len(datos_tr[0]); gen = torch.Generator().manual_seed(SEMILLA)
    mejor, estado, sin_mejora, ep_mejor = float("inf"), None, 0, 0
    for ep in range(1, epocas + 1):
        red.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, lote):
            j = perm[i:i + lote]
            opt.zero_grad()
            perdida(red(*[d[j] for d in datos_tr[:-1]]), datos_tr[-1][j]).backward()
            opt.step()
        red.eval()
        with torch.no_grad():
            v = perdida(red(*datos_va[:-1]), datos_va[-1]).item()
        if v < mejor - 1e-6:
            mejor, estado, sin_mejora, ep_mejor = v, {k: t.clone() for k, t in red.state_dict().items()}, 0, ep
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(estado); red.eval()
    print(f"  {etiqueta:<40} mejor epoca {ep_mejor:>3} de {ep}   perdida val {mejor:.4f}")
    return red


filas = []

# ==================================================== A. RED MULTICUANTIL
sec("A. RED MULTICUANTIL: LA FRONTERA CON UN SOLO MODELO")
print(f"Una red con {len(CUANTILES)} salidas, una por cuantil {CUANTILES},")
print("entrenada con una pinball combinada. Sustituye a nueve reentrenamientos.\n")

torch.manual_seed(SEMILLA)
multi = nn.Sequential(nn.Linear(len(FEATURES), 64), nn.ReLU(), nn.Dropout(0.10),
                      nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(CUANTILES)))
multi = entrenar(multi, pinball(CUANTILES), (Xtr, Ytr), (Xva, Yva), "Red multicuantil")
with torch.no_grad():
    Q = multi(Xte).numpy() * SIGMA
Q = np.clip(Q, 0, None)

cruces = float((np.diff(Q, axis=1) < 0).any(axis=1).mean() * 100)
print(f"\nCruce de cuantiles: {cruces:.1f} % de las observaciones de prueba presentan")
print("al menos un par de cuantiles desordenados. La red no impone monotonia,")
print("de modo que conviene declararlo como limitacion del enfoque.\n")

front = []
for k, a in enumerate(CUANTILES):
    rep = Q[:, k] * (1 + COLCHON)
    exc = float(np.clip(rep - y_te, 0, None).sum()); rot = float(np.clip(y_te - rep, 0, None).sum())
    front.append({"alpha": a, "excedente": exc, "rotura": rot,
                  "exc_pct": 100 * exc / DEMANDA, "rot_pct": 100 * rot / DEMANDA,
                  "MAE": mean_absolute_error(y_te, Q[:, k])})
    print(f"  cuantil {a:.2f}   excedente {exc:9,.0f} ({100*exc/DEMANDA:5.1f} %)"
          f"   rotura {rot:9,.0f} ({100*rot/DEMANDA:5.1f} %)   MAE {front[-1]['MAE']:.2f}")
fr = pd.DataFrame(front); fr.to_csv(RES / "frontera_red_multicuantil.csv", index=False)
filas.append(metricas("Red multicuantil (salida a=0,35)", Q[:, CUANTILES.index(0.35)]))

xgb_fr = pd.read_csv(RES / "frontera_excedente_rotura.csv")
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(xgb_fr.exc_pct, xgb_fr.rot_pct, "-o", color=AZUL, lw=1.8, ms=5,
        label="XGBoost cuantílico (nueve modelos)")
ax.plot(fr.exc_pct, fr.rot_pct, "-s", color=VERDE, lw=1.8, ms=5,
        label="Red multicuantil (un solo modelo)")
for _, r in fr.iterrows():
    ax.annotate(f"α={r.alpha:.2f}".replace(".", ","), (r.exc_pct, r.rot_pct),
                textcoords="offset points", xytext=(7, -11), fontsize=8.5, color=VERDE)
ax.set_xlabel("Excedente de reposición sobre la demanda (%)")
ax.set_ylabel("Rotura de stock sobre la demanda (%)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig16_frontera_multicuantil.png"); plt.close(fig)
print("\nFigura fig16 generada: las dos fronteras superpuestas.")

# ==================================================== B. RED RECURRENTE
sec("B. RED RECURRENTE (LSTM) SOBRE LA SECUENCIA CRUDA")
print(f"Entrada: las {VENTANA} semanas anteriores de demanda y de hogares activos.")
print("Sin retardos ni medias moviles: la red debe deducir la estructura temporal.\n")

seqs, obj, semanas = [], [], []
for c, d in panel.groupby("COMMODITY_DESC", observed=True):
    d = d.sort_values("WEEK_NO")
    u = d.unidades.values.astype(np.float32); h = d.hogares_activos.values.astype(np.float32)
    w = d.WEEK_NO.values
    for i in range(VENTANA, len(d)):
        seqs.append(np.stack([u[i - VENTANA:i], h[i - VENTANA:i]], axis=1))
        obj.append(u[i]); semanas.append(w[i])
S = np.log1p(np.array(seqs, dtype=np.float32)); O = np.array(obj, dtype=np.float32)
W = np.array(semanas)
m_tr, m_va, m_te = W < CORTE_VAL, (W >= CORTE_VAL) & (W < CORTE), W >= CORTE
mu, sd = S[m_tr].mean(axis=(0, 1)), S[m_tr].std(axis=(0, 1))
S = (S - mu) / sd
print(f"Ventanas: {m_tr.sum():,} entrenamiento / {m_va.sum():,} validacion / {m_te.sum():,} prueba")

Str, Sva, Ste = (torch.tensor(S[m]) for m in (m_tr, m_va, m_te))
Otr = torch.tensor(O[m_tr] / SIGMA).unsqueeze(1); Ova = torch.tensor(O[m_va] / SIGMA).unsqueeze(1)
y_te_lstm = O[m_te]


class Recurrente(nn.Module):
    def __init__(self, salidas=1):
        super().__init__()
        self.lstm = nn.LSTM(2, 32, batch_first=True)
        self.sal = nn.Linear(32, salidas)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.sal(o[:, -1, :])


def metricas_lstm(nombre, pred):
    pred = np.clip(pred, 0, None); m = y_te_lstm > 0
    rep = pred * (1 + COLCHON)
    exc = float(np.clip(rep - y_te_lstm, 0, None).sum()); rot = float(np.clip(y_te_lstm - rep, 0, None).sum())
    dem = float(y_te_lstm.sum())
    mae = mean_absolute_error(y_te_lstm, pred)
    return {"Modelo": nombre, "MAE": mae,
            "RMSE": float(np.sqrt(mean_squared_error(y_te_lstm, pred))),
            "MAPE (%)": float(np.mean(np.abs((y_te_lstm[m] - pred[m]) / y_te_lstm[m])) * 100),
            "Mejora sobre ingenuo (%)": 100 * (MAE_INGENUO - mae) / MAE_INGENUO,
            "Excedente (uds)": exc, "Rotura (uds)": rot,
            "Excedente sobre demanda (%)": 100 * exc / dem,
            "Reduccion del excedente (%)": 100 * (EXC_INGENUO - exc) / EXC_INGENUO}


for etiq, perd in [("LSTM (pérdida simétrica)", nn.MSELoss()),
                   ("LSTM cuantílico (α=0,35)", pinball([0.35]))]:
    torch.manual_seed(SEMILLA)
    red = entrenar(Recurrente(), perd, (Str, Otr), (Sva, Ova), etiq)
    with torch.no_grad():
        p = red(Ste).squeeze(1).numpy() * SIGMA
    filas.append(metricas_lstm(etiq, p))

print("\nAviso de comparabilidad: el LSTM usa una ventana de 13 semanas, de modo")
print("que su conjunto de prueba tiene las mismas semanas pero se construye de")
print("otra forma. Las cifras son comparables en orden de magnitud, no al decimal.")

# ==================================================== C. EMBEDDINGS
sec("C. EMBEDDINGS DE CATEGORIA")
print(f"Un vector aprendido de dimension 8 por cada una de las {len(cats)} categorias,")
print("concatenado a las 14 variables. Es una capacidad propia de las redes.\n")


class ConEmbedding(nn.Module):
    def __init__(self, n_cat, dim=8):
        super().__init__()
        self.emb = nn.Embedding(n_cat, dim)
        self.mlp = nn.Sequential(nn.Linear(len(FEATURES) + dim, 64), nn.ReLU(), nn.Dropout(0.10),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, idx):
        return self.mlp(torch.cat([x, self.emb(idx)], dim=1))


torch.manual_seed(SEMILLA)
rede = entrenar(ConEmbedding(len(cats)), pinball([0.35]),
                (Xtr, Itr, Ytr), (Xva, Iva, Yva), "Red con embeddings (α=0,35)")
with torch.no_grad():
    pe = rede(Xte, Ite).squeeze(1).numpy() * SIGMA
filas.append(metricas("Red con embeddings (α=0,35)", pe))

E = rede.emb.weight.detach().numpy()
proy = TSNE(n_components=2, perplexity=15, random_state=SEMILLA, init="pca").fit_transform(E)
vol = (panel.groupby("COMMODITY_DESC", observed=True).unidades
            .agg(lambda s: s.std() / s.mean()).reindex(cats).values)
volumen = panel.groupby("COMMODITY_DESC", observed=True).unidades.sum().reindex(cats).values

pd.DataFrame({"categoria": cats, "x": proy[:, 0], "y": proy[:, 1],
              "cv": vol, "volumen": volumen}).to_csv(RES / "embeddings_categorias.csv", index=False)

fig, ax = plt.subplots(figsize=(8.2, 6))
s = ax.scatter(proy[:, 0], proy[:, 1], c=vol, s=18 + 55 * volumen / volumen.max(),
               cmap="viridis_r", alpha=.85, edgecolors="white", linewidths=.4)
for i in np.argsort(-volumen)[:8]:
    ax.annotate(cats[i].title()[:22], (proy[i, 0], proy[i, 1]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5, color="#1a1a1a")
fig.colorbar(s, ax=ax, label="Coeficiente de variación de la categoría")
ax.set_xlabel("Dimensión 1 de la proyección"); ax.set_ylabel("Dimensión 2 de la proyección")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(FIG / "fig17_embeddings_categorias.png"); plt.close(fig)
print("\nFigura fig17 generada: proyección t-SNE de los embeddings.")
print("El tamaño del punto es el volumen de la categoría y el color su volatilidad.")

# ==================================================== resumen
sec("RESUMEN DE LOS TRES EXPERIMENTOS")
res = pd.DataFrame(filas)
pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:,.2f}")
print(res[["Modelo", "MAE", "RMSE", "MAPE (%)", "Mejora sobre ingenuo (%)"]].to_string(index=False))
print()
print(res[["Modelo", "Excedente (uds)", "Rotura (uds)",
           "Excedente sobre demanda (%)", "Reduccion del excedente (%)"]].to_string(index=False))
res.to_csv(RES / "redes_avanzadas.csv", index=False)
print(f"\nGuardado en {RES / 'redes_avanzadas.csv'}")
print(f"Frontera multicuantil en {RES / 'frontera_red_multicuantil.csv'}")
print(f"Figuras fig16 y fig17 en {FIG}/")
