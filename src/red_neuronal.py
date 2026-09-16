# -*- coding: utf-8 -*-
"""
Perceptron multicapa sobre el panel semana x categoria.

Se entrena en dos variantes que comparten arquitectura, datos y particion:
  - perdida simetrica (error cuadratico medio)
  - perdida cuantilica (pinball) en los mismos puntos de operacion que XGBoost

La comparacion entre ambas replica, en otra familia de modelos, el contraste
entre XGBoost estandar y XGBoost cuantilico del script modelado.py.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEMILLA = 42
CORTE = 88          # primera semana del conjunto de prueba
CORTE_VAL = 80      # las semanas 80-87 se reservan para la parada temprana
COLCHON = 0.10      # misma politica de reposicion que en modelado.py
PROC = Path("data/processed")
RES = Path("results")
RES.mkdir(exist_ok=True)

np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)


def sec(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


# --- panel y variables
sec("1. PANEL Y VARIABLES")

panel = (pd.read_parquet(PROC / "panel_modelado.parquet")
           .sort_values(["COMMODITY_DESC", "WEEK_NO"])
           .reset_index(drop=True))
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

FEATURES = [f"lag_{k}" for k in [1, 2, 3, 4]] + \
           [f"media_{w}" for w in [4, 8, 13]] + \
           ["desv_4", "tendencia_4", "ratio_lag1_media4", "nivel_categoria",
            "sin_ano", "cos_ano", "hogares_lag1"]

datos = panel.dropna(subset=FEATURES + ["unidades"]).copy()
print(f"Variables : {len(FEATURES)}")
print(f"Panel util: {len(datos):,} observaciones "
      f"(semanas {datos.WEEK_NO.min()} a {datos.WEEK_NO.max()})")

# --- particion temporal
sec("2. PARTICION TEMPORAL")

train = datos[datos.WEEK_NO < CORTE_VAL]
val = datos[(datos.WEEK_NO >= CORTE_VAL) & (datos.WEEK_NO < CORTE)]
test = datos[datos.WEEK_NO >= CORTE]

print(f"Entrenamiento : semanas {train.WEEK_NO.min()}-{train.WEEK_NO.max()}  {len(train):>6,} obs")
print(f"Validacion    : semanas {val.WEEK_NO.min()}-{val.WEEK_NO.max()}  {len(val):>6,} obs  (parada temprana)")
print(f"Prueba        : semanas {test.WEEK_NO.min()}-{test.WEEK_NO.max()}  {len(test):>6,} obs")
print("\nLa validacion es tambien temporal: se toman las ultimas semanas del")
print("entrenamiento, nunca semanas sueltas al azar, para no introducir fuga.")

# --- escalado (mismo criterio que el SVR)
sec("3. ESCALADO")

NIVEL = [f for f in FEATURES if f.startswith(("lag_", "media_", "desv_", "nivel_", "hogares_"))]
idx_nivel = [FEATURES.index(f) for f in NIVEL]


def a_log(X):
    Z = X.copy()
    Z[:, idx_nivel] = np.log1p(np.clip(Z[:, idx_nivel], 0, None))
    return Z


X_tr, y_tr = a_log(train[FEATURES].values), train.unidades.values.astype(np.float32)
X_va, y_va = a_log(val[FEATURES].values), val.unidades.values.astype(np.float32)
X_te, y_te = a_log(test[FEATURES].values), test.unidades.values.astype(np.float32)

esc = StandardScaler().fit(X_tr)
X_tr, X_va, X_te = esc.transform(X_tr), esc.transform(X_va), esc.transform(X_te)

# escalado lineal del objetivo: al ser lineal, conserva los cuantiles
SIGMA = float(y_tr.std())
print(f"Variables de nivel en logaritmo y estandarizadas, igual que el SVR.")
print(f"Objetivo dividido por su desviacion tipica ({SIGMA:.1f}) para estabilizar")
print("el entrenamiento. Al ser una transformacion lineal, el cuantil se conserva.")

T = lambda a: torch.tensor(a, dtype=torch.float32)
Xtr, Ytr = T(X_tr), T(y_tr / SIGMA).unsqueeze(1)
Xva, Yva = T(X_va), T(y_va / SIGMA).unsqueeze(1)
Xte = T(X_te)


# --- modelo y entrenamiento
def crear_red():
    torch.manual_seed(SEMILLA)
    return nn.Sequential(
        nn.Linear(len(FEATURES), 64), nn.ReLU(), nn.Dropout(0.10),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


def perdida_pinball(alpha):
    def f(pred, real):
        e = real - pred
        return torch.mean(torch.maximum(alpha * e, (alpha - 1) * e))
    return f


def entrenar(perdida, etiqueta, epocas=400, paciencia=40, lr=1e-3, lote=256):
    red = crear_red()
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    n = len(Xtr)
    mejor, mejor_estado, sin_mejora, epoca_mejor = float("inf"), None, 0, 0
    gen = torch.Generator().manual_seed(SEMILLA)

    for ep in range(1, epocas + 1):
        red.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, lote):
            j = perm[i:i + lote]
            opt.zero_grad()
            perdida(red(Xtr[j]), Ytr[j]).backward()
            opt.step()

        red.eval()
        with torch.no_grad():
            v = perdida(red(Xva), Yva).item()
        if v < mejor - 1e-6:
            mejor, mejor_estado, sin_mejora, epoca_mejor = v, \
                {k: t.clone() for k, t in red.state_dict().items()}, 0, ep
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    red.load_state_dict(mejor_estado)
    red.eval()
    with torch.no_grad():
        pred = red(Xte).squeeze(1).numpy() * SIGMA
    print(f"  {etiqueta:<34} mejor epoca {epoca_mejor:>3} de {ep}   perdida val {mejor:.4f}")
    return np.clip(pred, 0, None)


sec("4. ENTRENAMIENTO")
print("Arquitectura: 14 -> 64 -> 32 -> 1, activacion ReLU, dropout 0,10")
print("Optimizador Adam (lr=0,001), lotes de 256, parada temprana con paciencia 40.")
print("Las tres variantes comparten arquitectura, datos, particion y semilla:")
print("la unica diferencia entre ellas es la funcion de perdida.\n")

preds = {
    "Red neuronal (perdida simetrica)": entrenar(nn.MSELoss(), "Red neuronal (simetrica)"),
    "Red neuronal cuantilica (a=0.45)": entrenar(perdida_pinball(0.45), "Red neuronal cuantilica (a=0,45)"),
    "Red neuronal cuantilica (a=0.35)": entrenar(perdida_pinball(0.35), "Red neuronal cuantilica (a=0,35)"),
}

# --- metricas
sec("5. RESULTADOS SOBRE EL CONJUNTO DE PRUEBA")

MAE_INGENUO = 43.43524416135881          # linea base del script modelado.py
EXC_INGENUO = 88181.9                    # excedente de la politica ingenua


def evaluar(nombre, pred):
    mae = mean_absolute_error(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    m = y_te > 0
    mape = float(np.mean(np.abs((y_te[m] - pred[m]) / y_te[m])) * 100)
    rep = pred * (1 + COLCHON)
    exc = float(np.clip(rep - y_te, 0, None).sum())
    rot = float(np.clip(y_te - rep, 0, None).sum())
    return {
        "Modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape,
        "Mejora sobre ingenuo (%)": 100 * (MAE_INGENUO - mae) / MAE_INGENUO,
        "Excedente (uds)": exc, "Rotura (uds)": rot,
        "Excedente sobre demanda (%)": 100 * exc / y_te.sum(),
        "Reduccion del excedente (%)": 100 * (EXC_INGENUO - exc) / EXC_INGENUO,
    }


res = pd.DataFrame([evaluar(n, p) for n, p in preds.items()])
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:,.2f}")
print(res[["Modelo", "MAE", "RMSE", "MAPE (%)", "Mejora sobre ingenuo (%)"]].to_string(index=False))
print()
print(res[["Modelo", "Excedente (uds)", "Rotura (uds)",
           "Excedente sobre demanda (%)", "Reduccion del excedente (%)"]].to_string(index=False))

res.to_csv(RES / "red_neuronal.csv", index=False)
np.save(RES / "pred_red_neuronal.npy", np.vstack([preds[k] for k in preds]))

sec("RESUMEN")
print(f"Resultados guardados en {RES / 'red_neuronal.csv'}")
print("Comparar con results/comparativa_modelos.csv y results/impacto_excedente.csv")
