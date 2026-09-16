# -*- coding: utf-8 -*-
"""
Modelado de la demanda semanal por categoria.

Target      : unidades vendidas por categoria y semana, horizonte de una semana.
Validacion  : particion estrictamente temporal (nunca aleatoria).
Modelos     : dos lineas base ingenuas, XGBoost y SVR.
Salida      : metricas comparadas, importancia de variables, figuras y predicciones.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROC, FIG, RES = Path("data/processed"), Path("figures"), Path("results")
RES.mkdir(exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
AZUL, NARANJA, VERDE, ROJO, GRIS = "#2A6F97", "#E07A5F", "#4A7C59", "#C1121F", "#808080"


def sec(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


# --- datos
sec("1. PANEL DE PARTIDA")
panel = pd.read_parquet(PROC / "panel_modelado.parquet")
panel = panel.sort_values(["COMMODITY_DESC", "WEEK_NO"]).reset_index(drop=True)
print(f"Observaciones : {len(panel):,}")
print(f"Categorias    : {panel.COMMODITY_DESC.nunique()}")
print(f"Semanas       : {panel.WEEK_NO.min()} a {panel.WEEK_NO.max()}")

# --- ingenieria de variables
sec("2. INGENIERIA DE VARIABLES")
g = panel.groupby("COMMODITY_DESC", observed=True)

for k in [1, 2, 3, 4]:
    panel[f"lag_{k}"] = g.unidades.shift(k)
for w in [4, 8, 13]:
    panel[f"media_{w}"] = g.unidades.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
panel["desv_4"] = g.unidades.shift(1).rolling(4).std().reset_index(level=0, drop=True)

# dinamica reciente
panel["tendencia_4"] = panel["media_4"] - panel["lag_1"]
panel["ratio_lag1_media4"] = panel["lag_1"] / panel["media_4"].replace(0, np.nan)

# contexto de la categoria y del calendario relativo
panel["nivel_categoria"] = g.unidades.transform(lambda s: s.shift(1).expanding().mean())
panel["semana_del_ano"] = panel.WEEK_NO % 52
panel["sin_ano"] = np.sin(2 * np.pi * panel.semana_del_ano / 52)
panel["cos_ano"] = np.cos(2 * np.pi * panel.semana_del_ano / 52)

# actividad del panel: conocida a priori, no introduce fuga
panel["hogares_lag1"] = g.hogares_activos.shift(1)

FEATURES = [f"lag_{k}" for k in [1, 2, 3, 4]] + \
           [f"media_{w}" for w in [4, 8, 13]] + \
           ["desv_4", "tendencia_4", "ratio_lag1_media4", "nivel_categoria",
            "sin_ano", "cos_ano", "hogares_lag1"]

datos = panel.dropna(subset=FEATURES + ["unidades"]).copy()
print(f"Variables construidas : {len(FEATURES)}")
print(f"Observaciones utiles  : {len(datos):,} (se pierden las de arranque por los retardos)")
print(f"Rango temporal util   : semanas {datos.WEEK_NO.min()} a {datos.WEEK_NO.max()}")

# --- particion temporal
sec("3. PARTICION TEMPORAL")
CORTE = 88
train = datos[datos.WEEK_NO < CORTE]
test = datos[datos.WEEK_NO >= CORTE]
print(f"Entrenamiento : semanas {train.WEEK_NO.min()} a {train.WEEK_NO.max()}  ->  {len(train):,} obs")
print(f"Prueba        : semanas {test.WEEK_NO.min()} a {test.WEEK_NO.max()}  ->  {len(test):,} obs")
print("La particion es temporal: el conjunto de prueba es siempre posterior al de entrenamiento.")

X_tr, y_tr = train[FEATURES].values, train.unidades.values
X_te, y_te = test[FEATURES].values, test.unidades.values


def metricas(nombre, real, pred):
    pred = np.clip(pred, 0, None)
    mae = mean_absolute_error(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))
    mask = real > 0
    mape = np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100
    return {"Modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


resultados, predicciones = [], {}

# --- lineas base
sec("4. LINEAS BASE")
p_naive = test.lag_1.values
resultados.append(metricas("Ingenuo (semana anterior)", y_te, p_naive))
predicciones["Ingenuo"] = p_naive

p_media = test.media_4.values
resultados.append(metricas("Media movil de 4 semanas", y_te, p_media))
predicciones["Media movil"] = p_media

for r in resultados:
    print(f"  {r['Modelo']:<28} MAE={r['MAE']:8.2f}  RMSE={r['RMSE']:8.2f}  MAPE={r['MAPE (%)']:6.2f}%")

# --- XGBoost
sec("5. XGBOOST")
xgb = XGBRegressor(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist",
)
xgb.fit(X_tr, y_tr)
p_xgb = xgb.predict(X_te)
resultados.append(metricas("XGBoost", y_te, p_xgb))
predicciones["XGBoost"] = p_xgb
r = resultados[-1]
print(f"  MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  MAPE={r['MAPE (%)']:.2f}%")

# --- SVR
sec("6. MAQUINA DE VECTORES SOPORTE")
# el SVR exige escalado; la demanda es muy asimetrica, se entrena sobre el logaritmo
NIVEL = [f for f in FEATURES if f.startswith(("lag_", "media_", "desv_", "nivel_", "hogares_"))]
idx_nivel = [FEATURES.index(f) for f in NIVEL]


def a_log(X):
    """Las variables de nivel deben ir en la misma escala logaritmica que el objetivo."""
    Z = X.copy()
    Z[:, idx_nivel] = np.log1p(np.clip(Z[:, idx_nivel], 0, None))
    return Z


esc = StandardScaler().fit(a_log(X_tr))
Xs_tr, Xs_te = esc.transform(a_log(X_tr)), esc.transform(a_log(X_te))
svr = SVR(kernel="rbf", C=3.0, gamma="scale", epsilon=0.1)
svr.fit(Xs_tr, np.log1p(y_tr))
p_svr = np.expm1(svr.predict(Xs_te))
resultados.append(metricas("SVR (nucleo RBF)", y_te, p_svr))
predicciones["SVR"] = p_svr
r = resultados[-1]
print(f"  MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  MAPE={r['MAPE (%)']:.2f}%")

# --- XGBoost con perdida asimetrica
sec("6b. XGBOOST CUANTILICO (perdida asimetrica)")
print("El error absoluto penaliza igual pasarse que quedarse corto, pero para reducir")
print("desperdicio pasarse es peor: la unidad sobrante se convierte en merma. Se entrena")
print("regresion cuantilica con cuantiles por debajo de la mediana para inclinar la")
print("prediccion a la baja de forma controlada.\n")

for alpha in [0.35, 0.45]:
    q = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    q.fit(X_tr, y_tr)
    pq = q.predict(X_te)
    nombre = f"XGBoost cuantilico (a={alpha})"
    resultados.append(metricas(nombre, y_te, pq))
    predicciones[nombre] = pq
    r = resultados[-1]
    print(f"  alpha={alpha}:  MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  MAPE={r['MAPE (%)']:.2f}%")

# --- comparativa
sec("7. COMPARATIVA")
tab = pd.DataFrame(resultados)
base_mae = tab.loc[tab.Modelo == "Ingenuo (semana anterior)", "MAE"].iloc[0]
tab["Mejora sobre ingenuo (%)"] = 100 * (base_mae - tab.MAE) / base_mae
tab = tab.sort_values("MAE").reset_index(drop=True)
print(tab.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
tab.to_csv(RES / "comparativa_modelos.csv", index=False)

mejor = tab.iloc[0].Modelo
print(f"\nMejor modelo: {mejor}")

# --- importancia variables
sec("8. IMPORTANCIA DE VARIABLES (XGBoost)")
imp = (pd.DataFrame({"variable": FEATURES, "importancia": xgb.feature_importances_})
       .sort_values("importancia", ascending=False).reset_index(drop=True))
print(imp.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
imp.to_csv(RES / "importancia_variables.csv", index=False)

# --- figuras
sec("9. FIGURAS")

# comparativa de error
fig, ax = plt.subplots(figsize=(8, 3.8))
orden = tab.sort_values("MAE")
colores = [VERDE if m == mejor else (GRIS if "Ingenuo" in m or "Media" in m else AZUL)
           for m in orden.Modelo]
ax.barh(orden.Modelo, orden.MAE, color=colores, height=0.6)
for i, (m, v) in enumerate(zip(orden.Modelo, orden.MAE)):
    ax.text(v + orden.MAE.max() * 0.015, i, f"{v:,.1f}".replace(",", "."), va="center", fontsize=9)
ax.set_xlabel("Error absoluto medio (unidades)")
ax.invert_yaxis()
ax.set_xlim(0, orden.MAE.max() * 1.15)
fig.tight_layout()
fig.savefig(FIG / "fig06_comparativa_modelos.png")
plt.close(fig)

# importancia
fig, ax = plt.subplots(figsize=(8, 4.2))
top = imp.head(12).iloc[::-1]
nombres = {
    "lag_1": "Demanda semana anterior", "lag_2": "Demanda 2 semanas antes",
    "lag_3": "Demanda 3 semanas antes", "lag_4": "Demanda 4 semanas antes",
    "media_4": "Media de 4 semanas",
    "media_8": "Media de 8 semanas", "media_13": "Media de 13 semanas",
    "desv_4": "Desviación de 4 semanas", "tendencia_4": "Tendencia reciente",
    "ratio_lag1_media4": "Cociente último valor / media", "nivel_categoria": "Nivel medio de la categoría",
    "sin_ano": "Estacionalidad (seno)", "cos_ano": "Estacionalidad (coseno)",
    "hogares_lag1": "Hogares activos previos",
}
ax.barh([nombres.get(v, v) for v in top.variable], top.importancia, color=AZUL, height=0.65)
ax.set_xlabel("Importancia relativa")
fig.tight_layout()
fig.savefig(FIG / "fig07_importancia_variables.png")
plt.close(fig)

# ajuste sobre categorias representativas
test_pred = test.copy()
test_pred["pred"] = np.clip(predicciones["XGBoost"], 0, None)
top4 = test_pred.groupby("COMMODITY_DESC").unidades.sum().sort_values(ascending=False).head(4).index
etiquetas = {"SOFT DRINKS": "Refrescos", "FLUID MILK PRODUCTS": "Leche", "CHEESE": "Queso",
             "BAG SNACKS": "Aperitivos", "BAKED BREAD/BUNS/ROLLS": "Pan", "BEEF": "Vacuno",
             "SOUP": "Sopa", "FROZEN PIZZA": "Pizza congelada"}
fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
for a, c in zip(axes.ravel(), top4):
    s = test_pred[test_pred.COMMODITY_DESC == c].sort_values("WEEK_NO")
    a.plot(s.WEEK_NO, s.unidades, lw=1.8, color=AZUL, label="Real")
    a.plot(s.WEEK_NO, s.pred, lw=1.8, color=NARANJA, ls="--", label="Predicción")
    a.set_title(etiquetas.get(c, c.title()), fontsize=10)
    a.set_xlabel("Semana")
    a.set_ylabel("Unidades")
axes.ravel()[0].legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig08_ajuste_categorias.png")
plt.close(fig)

# --- traduccion a desperdicio
sec("10. IMPACTO SOBRE EL EXCEDENTE DE REPOSICION")
# Politica de reposicion: reponer la prediccion mas un colchon de seguridad.
# El excedente es lo repuesto por encima de lo vendido; la rotura es lo contrario.
COLCHON = 0.10
resumen_imp = []
pares = [("Ingenuo (semana anterior)", "Ingenuo"), ("Media movil de 4 semanas", "Media movil"),
         ("XGBoost", "XGBoost"), ("SVR (nucleo RBF)", "SVR")]
pares += [(k, k) for k in predicciones if k.startswith("XGBoost cuantilico")]
for nombre, clave in pares:
    pred = predicciones[clave]
    repuesto = np.clip(pred, 0, None) * (1 + COLCHON)
    excedente = np.clip(repuesto - y_te, 0, None).sum()
    rotura = np.clip(y_te - repuesto, 0, None).sum()
    resumen_imp.append({"Modelo": nombre, "Excedente (uds)": excedente,
                        "Rotura (uds)": rotura, "Excedente sobre demanda (%)": 100 * excedente / y_te.sum()})

imp_tab = pd.DataFrame(resumen_imp).sort_values("Excedente (uds)").reset_index(drop=True)
base_exc = imp_tab.loc[imp_tab.Modelo == "Ingenuo (semana anterior)", "Excedente (uds)"].iloc[0]
imp_tab["Reduccion del excedente (%)"] = 100 * (base_exc - imp_tab["Excedente (uds)"]) / base_exc
print(f"Demanda total del periodo de prueba: {y_te.sum():,.0f} unidades")
print(f"Politica simulada: reponer la prediccion con un colchon del {COLCHON:.0%}\n")
print(imp_tab.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
imp_tab.to_csv(RES / "impacto_excedente.csv", index=False)

test_pred.to_parquet(RES / "predicciones_test.parquet", index=False)

# --- frontera entre excedente y rotura de stock
sec("11. FRONTERA ENTRE EXCEDENTE Y ROTURA")
print("Se recorre el cuantil objetivo para trazar el compromiso entre merma y rotura.\n")

frontera = []
for alpha in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
    q = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    q.fit(X_tr, y_tr)
    rep = np.clip(q.predict(X_te), 0, None) * (1 + COLCHON)
    exc = np.clip(rep - y_te, 0, None).sum()
    rot = np.clip(y_te - rep, 0, None).sum()
    frontera.append({"alpha": alpha, "excedente": exc, "rotura": rot,
                     "exc_pct": 100 * exc / y_te.sum(), "rot_pct": 100 * rot / y_te.sum()})
    print(f"  alpha={alpha:.2f}  excedente={exc:9,.0f} ({100*exc/y_te.sum():5.1f}%)"
          f"   rotura={rot:9,.0f} ({100*rot/y_te.sum():5.1f}%)")

fr = pd.DataFrame(frontera)
fr.to_csv(RES / "frontera_excedente_rotura.csv", index=False)

# referencias de las politicas de partida
ref = []
for nom, cl in [("Ingenuo", "Ingenuo"), ("Media móvil", "Media movil")]:
    rep = np.clip(predicciones[cl], 0, None) * (1 + COLCHON)
    ref.append((nom, 100 * np.clip(rep - y_te, 0, None).sum() / y_te.sum(),
                100 * np.clip(y_te - rep, 0, None).sum() / y_te.sum()))

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(fr.exc_pct, fr.rot_pct, "-o", color=AZUL, lw=1.8, ms=5, label="Modelo cuantílico")
for _, r in fr.iterrows():
    if r.alpha in (0.20, 0.35, 0.50, 0.70):
        ax.annotate(f"α={r.alpha:.2f}".replace(".", ","), (r.exc_pct, r.rot_pct),
                    textcoords="offset points", xytext=(7, 6), fontsize=8.5, color=AZUL)
for nom, e, ro in ref:
    ax.scatter([e], [ro], s=70, marker="s", color=ROJO, zorder=5)
    ax.annotate(nom, (e, ro), textcoords="offset points", xytext=(8, -12), fontsize=9, color=ROJO)
ax.set_xlabel("Excedente de reposición sobre la demanda (%)")
ax.set_ylabel("Rotura de stock sobre la demanda (%)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig09_frontera_excedente_rotura.png")
plt.close(fig)
print("\nFigura fig09 generada: frontera entre excedente y rotura.")

sec("RESUMEN")
print(f"Mejor modelo por error   : {mejor}")
print(f"Figuras nuevas           : fig06, fig07, fig08")
print(f"Resultados guardados en  : results/")
