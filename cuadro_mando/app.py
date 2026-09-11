# -*- coding: utf-8 -*-
"""
Cuadro de mando para la decision semanal de reposicion.

El destinatario es un responsable de tienda o de categoria, no un analista.
De ahi las decisiones que gobiernan el diseno, justificadas en el apartado
7.2 de la memoria:

  - No se pregunta por el cuantil objetivo, que es jerga. Se pregunta por
    los tres costes del negocio y el punto de operacion se deduce de ahi.
    El apartado 2.1 deja esa eleccion "en manos de los expertos que conocen
    con detalle todos sus costes", asi que fijarlos en el codigo seria
    contradecir el objetivo del trabajo.
  - Cada categoria se valora a su precio medio real, calculado del propio
    dato. Sumar unidades de vino y de yogur no significa nada.
  - El producto perecedero y el que aguanta reciben puntos de operacion
    distintos. Si sobra una lata se vende la semana siguiente; si sobra un
    filete se tira. Tratarlos igual sobreestima el desperdicio.
  - El nivel por categoria lo ajusta el usuario, no el algoritmo. Con
    quince semanas de prueba por categoria, el nivel que sale mejor en la
    primera mitad del periodo solo coincide con el de la segunda en el
    10 % de los casos, y ajustar asi sale un 7,6 % mas caro que aplicar el
    nivel del grupo.

La navegacion es por pestanas y no por desplegables: cada una es una
pregunta distinta y solo la primera depende de la semana elegida.

No se entrena ningun modelo: se leen las predicciones ya calculadas por
src/datos_cuadro_mando.py.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATOS = Path(__file__).parent / "datos"
AZUL, VERDE, NARANJA, GRIS = "#2A6F97", "#4A7C59", "#E07A5F", "#6C757D"

st.set_page_config(page_title="Cuánto reponer esta semana", page_icon="🛒",
                   layout="wide")


@st.cache_data
def cargar():
    pred = pd.read_parquet(DATOS / "predicciones_cuantiles.parquet")
    cat = pd.read_parquet(DATOS / "categorias.parquet")
    hist = pd.read_parquet(DATOS / "historico.parquet")
    ref = pd.read_parquet(DATOS / "referencias.parquet")
    pred = pred.merge(cat, on="COMMODITY_DESC", how="left")
    pred["regimen"] = np.where(pred.perecedero, "Perecedero", "Aguanta")
    return pred, hist, cat, ref


base, hist, cat, referencias = cargar()
QS = sorted(int(c[1:]) for c in base.columns if c.startswith("q") and c[1:].isdigit())
SEMANAS = sorted(base.WEEK_NO.unique())
CATEGORIAS = sorted(base.COMMODITY_DESC.unique())
REGIMENES = ["Perecedero", "Aguanta"]
TITULO = {"Perecedero": "Fresco", "Aguanta": "Lo que aguanta"}
COLOR = {"Perecedero": NARANJA, "Aguanta": AZUL}


def eur(x, dec=0):
    s = f"{x:,.{dec}f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return f"{s} €"


def num(x):
    return f"{x:,.0f}".replace(",", ".")


def bonito(n):
    return n.title().replace("/", " / ")


def signo(x):
    return "+" if x > 0 else "-"


def dec(x, n=1):
    return f"{x:.{n}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def curvas_coste(cs_pere, cs_agua, cf):
    """Coste total de cada nivel de cobertura, separado por regimen.

    Se cachea porque solo depende de los tres costes de la barra lateral,
    no de la semana ni de la categoria que el usuario tenga elegidas.
    """
    cs = {"Perecedero": cs_pere, "Aguanta": cs_agua}
    filas = []
    for r in REGIMENES:
        s = base[base.regimen == r]
        real, pvp = s.unidades.values, s.precio.values
        for q in QS:
            rep = s[f"q{q:02d}"].values
            exc = np.clip(rep - real, 0, None)
            rot = np.clip(real - rep, 0, None)
            filas.append({"regimen": r, "Cuantil": q,
                          "exc_eur": float((exc * pvp * cs[r]).sum()),
                          "rot_eur": float((rot * pvp * cf).sum())})
    c = pd.DataFrame(filas)
    c["coste"] = c.exc_eur + c.rot_eur
    return c


@st.cache_data(show_spinner=False)
def cuantil_unico(cs_pere, cs_agua, cf):
    """Coste en euros de aplicar un mismo cuantil a todo el surtido.

    Es la comparacion justa para la politica de dos grupos: si separar por
    perecibilidad no aportara nada, el mejor cuantil unico saldria igual de
    barato.
    """
    real, pvp = base.unidades.values, base.precio.values
    cs = np.where(base.perecedero.values, cs_pere, cs_agua)
    filas = []
    for q in QS:
        rep = base[f"q{q:02d}"].values
        exc, rot = np.clip(rep - real, 0, None), np.clip(real - rep, 0, None)
        filas.append({"Cuantil": q, "coste": float((exc * pvp * cs + rot * pvp * cf).sum())})
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False)
def frontera():
    """Frontera excedente-rotura recorriendo el cuantil objetivo.

    Un unico cuantil para todas las categorias, que es como se traza en el
    capitulo 6. Aqui va sin el colchon de seguridad del 10 % que se aplica
    alli, porque sumar un colchon a una prediccion de cuantil duplica el
    margen: el cuantil ya codifica el nivel de servicio. Por eso las cifras
    absolutas no coinciden con las de la figura 16, aunque la forma si.
    """
    real = base.unidades.values
    dem = real.sum()
    filas = []
    for q in QS:
        rep = base[f"q{q:02d}"].values
        filas.append({"Cuantil": q,
                      "exc_pct": 100 * np.clip(rep - real, 0, None).sum() / dem,
                      "rot_pct": 100 * np.clip(real - rep, 0, None).sum() / dem})
    return pd.DataFrame(filas)


# ================================================= barra lateral: los costes
st.sidebar.title("Tus costes")
st.sidebar.caption("La herramienta no decide por ti. Tú pones lo que te cuesta "
                   "cada error y de ahí sale cuánto conviene pedir.")

m = st.sidebar.slider("Margen bruto medio (%)", 5, 60, 30, 1,
                      help="Qué parte del precio de venta te queda como margen. "
                           "En alimentación suele estar entre el 20 % y el 35 %.") / 100
rec = st.sidebar.slider("De lo que no vendes, ¿cuánto recuperas? (%)", 0, 80, 0, 5,
                        help="Con liquidaciones, descuentos de última hora o "
                             "donaciones con desgravación. Si no rescatas nada, "
                             "déjalo en cero, que es el caso conservador.") / 100
phi = st.sidebar.slider("Del sobrante que aguanta, ¿qué parte pierdes? (%)", 0, 100, 15, 5,
                        help="Si te pasas pidiendo conservas o refrescos, los vendes "
                             "la semana siguiente. Esto es la parte que aun así acabas "
                             "perdiendo por roturas, caducidad u obsolescencia.") / 100

c_sobra = {"Perecedero": (1 - m) * (1 - rec), "Aguanta": (1 - m) * (1 - rec) * phi}
c_falta = m
curvas = curvas_coste(c_sobra["Perecedero"], c_sobra["Aguanta"], c_falta)
optimo = {r: int(curvas[curvas.regimen == r].sort_values("coste").iloc[0].Cuantil)
          for r in REGIMENES}

st.sidebar.divider()
st.sidebar.markdown("#### Cuánto cubrir")
st.sidebar.markdown(f"**Fresco:** el **{optimo['Perecedero']} %** de las semanas")
st.sidebar.markdown(f"**Lo que aguanta:** el **{optimo['Aguanta']} %**")
st.sidebar.caption("Al fresco conviene pedirle justo, porque lo que sobra se tira. "
                   "A lo que aguanta conviene pedirle de más, porque lo que sobra "
                   "se vende la semana siguiente y quedarse corto sí cuesta margen.")

# ==================================================== decisión y resultados
if "ajustes" not in st.session_state:
    st.session_state.ajustes = {}

d = base.copy()
d["c_sobra"] = d.regimen.map(c_sobra)
d["nivel_base"] = d.regimen.map(optimo)
d["nivel"] = (d.COMMODITY_DESC.map(st.session_state.ajustes)
               .fillna(d.nivel_base).astype(int))


def repone(df, col_nivel):
    """Cantidad a reponer segun el nivel de cobertura de cada fila."""
    qs = sorted(df[col_nivel].unique())
    return np.select([df[col_nivel].values == q for q in qs],
                     [df[f"q{q:02d}"].values for q in qs], default=np.nan)


d["pedido"] = repone(d, "nivel")
d["sin_ajustes"] = repone(d, "nivel_base")
for nom in ["pedido", "sin_ajustes", "ingenua"]:
    rep = d[nom].clip(lower=0)
    d[f"exc_{nom}"] = (rep - d.unidades).clip(lower=0)
    d[f"rot_{nom}"] = (d.unidades - rep).clip(lower=0)
    d[f"coste_{nom}"] = (d[f"exc_{nom}"] * d.precio * d.c_sobra
                         + d[f"rot_{nom}"] * d.precio * c_falta)

coste_pedido, coste_ingenua = d.coste_pedido.sum(), d.coste_ingenua.sum()
ahorro = coste_ingenua - coste_pedido

# ================================================================= cabecera
st.title("Cuánto reponer esta semana")
st.caption("Cuánto conviene pedir de cada categoría, en unidades y en euros. "
           "La cantidad no es la venta más probable: es la que sale más barata "
           "una vez contado lo que cuesta cada tipo de fallo.")

t1, t2, t3, t4, t5 = st.tabs(["  El pedido  ", "  Por qué estas cantidades  ",
                              "  La frontera  ", "  Ajustar por categoría  ",
                              "  Ver una categoría  "])

# ------------------------------------------------------------ 1. el pedido
with t1:
    semana = st.select_slider("Semana a preparar", options=SEMANAS, value=SEMANAS[0],
                              help="Solo esta pestaña depende de la semana. Las "
                                   "otras tres resumen las quince juntas.")

    s = d[d.WEEK_NO == semana].copy()
    s["Categoría"] = s.COMMODITY_DESC.map(bonito)
    s["Tipo"] = s.regimen
    s["Cobertura"] = s.nivel
    s["Pedir"] = np.ceil(s.pedido).astype(int)
    s["Valor del pedido"] = (s["Pedir"] * s.precio).round(0)
    s["Vendido de verdad"] = s.unidades.astype(int)
    s["Sobró"] = np.floor(s.exc_pedido).astype(int)
    s["Faltó"] = np.floor(s.rot_pedido).astype(int)
    s = s.sort_values("Valor del pedido", ascending=False).reset_index(drop=True)

    a, b, c = st.columns(3)
    a.metric("Categorías en el pedido", num(len(s)))
    b.metric("Unidades a pedir", num(s["Pedir"].sum()))
    c.metric("Valor del pedido", eur(s["Valor del pedido"].sum()))

    st.dataframe(s[["Categoría", "Tipo", "Cobertura", "Pedir", "Valor del pedido"]],
                 width="stretch", hide_index=True, height=400,
                 column_config={
                     "Valor del pedido": st.column_config.NumberColumn(format="%.0f €"),
                     "Cobertura": st.column_config.NumberColumn(
                         format="%d %%",
                         help="Qué parte de las semanas quedaría cubierta con esa "
                              "cantidad. Se cambia en «Ajustar por categoría».")})
    st.download_button("Descargar el pedido en CSV",
                       s[["Categoría", "Tipo", "Pedir"]].to_csv(index=False).encode("utf-8"),
                       file_name=f"pedido_semana_{semana}.csv", mime="text/csv")

    st.divider()
    st.subheader(f"Qué pasó de verdad en la semana {semana}")
    st.caption("Al cursar el pedido esto no se sabe todavía. Está para poder "
               "comprobar si la recomendación acertó.")
    e, f = st.columns(2)
    e.metric("Coste de los fallos con este pedido", eur(s.coste_pedido.sum()))
    dif = s.coste_ingenua.sum() - s.coste_pedido.sum()
    f.metric("Frente a repetir la semana pasada", eur(abs(dif)),
             delta="ahorras" if dif > 0 else "pierdes",
             delta_color="normal" if dif > 0 else "inverse")
    st.dataframe(s[["Categoría", "Pedir", "Vendido de verdad", "Sobró", "Faltó"]],
                 width="stretch", hide_index=True, height=300)
    por_semana = d.groupby("WEEK_NO")[["coste_pedido", "coste_ingenua"]].sum()
    gana = int((por_semana.coste_ingenua > por_semana.coste_pedido).sum())
    st.caption(f"Hay semanas sueltas en las que esta forma de pedir sale peor. "
               f"En el conjunto de las {len(por_semana)} gana en {gana}.")

# --------------------------------------------------------------- 2. por qué
with t2:
    st.subheader("El fresco y el resto no se piden igual")
    st.markdown(
        f"Con un margen del **{m:.0%}**, cada unidad que falta te hace perder ese "
        f"margen, la tires o no. Lo que cambia es el otro lado. Si sobra fresco, "
        f"pierdes **{c_sobra['Perecedero']:.0%} del precio de venta**. Si sobra "
        f"producto que aguanta, solo **{c_sobra['Aguanta']:.0%}**, porque casi todo "
        "se acaba vendiendo. Por eso al fresco conviene pedirle justo y al resto "
        "de más. Cada curva tiene su propio mínimo, marcado en verde, y están en "
        "lados opuestos.")

    for caja, r in zip(st.columns(2), REGIMENES):
        cur = curvas[curvas.regimen == r]
        linea = alt.Chart(cur).mark_line(color=COLOR[r], strokeWidth=2.5, point=True).encode(
            x=alt.X("Cuantil:Q", title="Cobertura (%)",
                    scale=alt.Scale(zero=False, nice=True)),
            y=alt.Y("coste:Q", title="Coste (€)",
                    scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("Cuantil", title="Cubre el (%)"),
                     alt.Tooltip("coste", title="Coste (€)", format=",.0f")])
        punto = alt.Chart(cur[cur.Cuantil == optimo[r]]).mark_point(
            color=VERDE, size=300, filled=True).encode(x="Cuantil:Q", y="coste:Q")
        caja.markdown(f"**{TITULO[r]}** · cubrir el **{optimo[r]} %**")
        caja.altair_chart((linea + punto).properties(height=280), width="stretch")

    st.divider()
    st.subheader("Frente a repetir el pedido de la semana pasada")
    # El agregado de unidades engana: la recomendacion mueve sobrante entre
    # los dos tipos de producto, asi que en total puede sobrar mas y tirarse
    # mucho menos. Por eso la tabla va desglosada y no agregada.
    pe, ag = d[d.perecedero], d[~d.perecedero]
    comparativa = pd.DataFrame({
        "": ["Sobra en fresco (se tira)", "Falta en fresco",
             "Sobra en lo que aguanta", "Falta en lo que aguanta",
             "Coste de los fallos"],
        "Con la recomendación": [f"{num(pe.exc_pedido.sum())} uds",
                                 f"{num(pe.rot_pedido.sum())} uds",
                                 f"{num(ag.exc_pedido.sum())} uds",
                                 f"{num(ag.rot_pedido.sum())} uds",
                                 eur(coste_pedido)],
        "Repitiendo la semana pasada": [f"{num(pe.exc_ingenua.sum())} uds",
                                        f"{num(pe.rot_ingenua.sum())} uds",
                                        f"{num(ag.exc_ingenua.sum())} uds",
                                        f"{num(ag.rot_ingenua.sum())} uds",
                                        eur(coste_ingenua)]})
    st.dataframe(comparativa, width="stretch", hide_index=True)

    def var(a, b):
        return 100 * (a - b) / b if b else 0.0

    v_pe = var(pe.exc_pedido.sum(), pe.exc_ingenua.sum())
    v_ag = var(ag.exc_pedido.sum(), ag.exc_ingenua.sum())
    v_rot = var(d.rot_pedido.sum(), d.rot_ingenua.sum())
    mov = lambda v: "sube" if v > 0 else "baja"
    caja = st.success if v_pe < 0 else st.warning
    caja(f"**El sobrante de fresco, que es el que acaba en la basura, "
         f"{mov(v_pe)} un {abs(v_pe):.0f} %.** El de producto que aguanta, que se "
         f"vende la semana siguiente, {mov(v_ag)} un {abs(v_ag):.0f} %, y la rotura "
         f"total {mov(v_rot)} un {abs(v_rot):.0f} %. Lee las filas por separado: "
         "sumar unidades mezcla lo que se tira con lo que solo se adelanta una "
         "semana, y así el total puede moverse en sentido contrario al desperdicio "
         "de verdad.", icon="✅" if v_pe < 0 else "⚠️")
    st.caption(f"Suma de las {len(SEMANAS)} semanas de prueba. En euros el ahorro es "
               f"de {eur(ahorro)}, un {100 * ahorro / coste_ingenua:.0f} %."
               .replace(".0 %", " %"))
    st.info(
        "**El euro no es comparable entre supuestos distintos.** Está medido con "
        "los costes que has declarado a la izquierda. Si declaras que cada error "
        "te cuesta más, el ahorro sube, pero no porque el método mejore: sube "
        "porque la política de referencia comete más fallos y por tanto se "
        "encarece más deprisa. Es la vara de medir la que crece. Para comparar un "
        "supuesto con otro mira las unidades, que describen lo que pasa en la "
        "tienda: subir la pérdida del sobrante hasta el 100 % baja el excedente "
        "pero dispara la rotura, que es moverse por la frontera y no mejorar.",
        icon="ℹ️")

# --------------------------------------------------------- 3. la frontera
with t3:
    st.subheader("No hay una cantidad buena en abstracto")
    st.markdown(
        "Reducir el excedente siempre encarece la rotura. No existe un punto "
        "óptimo universal, existe una **frontera**: el conjunto de compromisos "
        "que se pueden alcanzar. Cada punto de la curva es un nivel de cobertura "
        "distinto. Bajar por ella significa pedir menos y tirar menos, a cambio "
        "de quedarte corto más veces. Tus costes son los que eligen en qué punto "
        "te conviene estar, y por eso la elección es tuya y no del modelo.")

    fr = frontera()
    dem = d.unidades.sum()
    actual = pd.DataFrame([{
        "modelo": "Donde estás ahora",
        "exc_pct": 100 * d.exc_pedido.sum() / dem,
        "rot_pct": 100 * d.rot_pedido.sum() / dem}])

    curva = alt.Chart(fr).mark_line(color=AZUL, strokeWidth=2.5, point=alt.OverlayMarkDef(
        color=AZUL, size=45)).encode(
        x=alt.X("exc_pct:Q", title="Excedente sobre la demanda (%)",
                scale=alt.Scale(zero=False, nice=True)),
        y=alt.Y("rot_pct:Q", title="Rotura sobre la demanda (%)",
                scale=alt.Scale(zero=False, nice=True)),
        tooltip=[alt.Tooltip("Cuantil", title="Cobertura (%)"),
                 alt.Tooltip("exc_pct", title="Excedente (%)", format=".2f"),
                 alt.Tooltip("rot_pct", title="Rotura (%)", format=".2f")])

    refs = alt.Chart(referencias).mark_point(
        shape="square", size=110, filled=True, color=NARANJA).encode(
        x="exc_pct:Q", y="rot_pct:Q",
        tooltip=["modelo", alt.Tooltip("exc_pct", title="Excedente (%)", format=".2f"),
                 alt.Tooltip("rot_pct", title="Rotura (%)", format=".2f")])
    # Los tres puntos de referencia caen muy juntos, asi que la etiqueta de
    # cada uno lleva su propio desplazamiento. Con un desplazamiento comun se
    # solapan entre ellas y con el punto del usuario. El gris medio se lee
    # igual sobre fondo claro que sobre fondo oscuro.
    POS = {"Repetir la semana pasada": (10, -11, "left"),
           "Media móvil de 4 semanas": (-11, -7, "right"),
           "XGBoost sin asimetría": (-11, 15, "right")}
    etiq = alt.layer(*[
        alt.Chart(referencias[referencias.modelo == mod]).mark_text(
            align=al, dx=dx, dy=dy, fontSize=11, color="#5a6472").encode(
            x="exc_pct:Q", y="rot_pct:Q", text="modelo")
        for mod, (dx, dy, al) in POS.items()])

    aqui = alt.Chart(actual).mark_point(
        shape="triangle", size=320, filled=True, color=VERDE).encode(
        x="exc_pct:Q", y="rot_pct:Q",
        tooltip=["modelo", alt.Tooltip("exc_pct", title="Excedente (%)", format=".2f"),
                 alt.Tooltip("rot_pct", title="Rotura (%)", format=".2f")])
    etiq_aqui = alt.Chart(actual).mark_text(
        align="left", dx=14, dy=8, fontSize=12, fontWeight="bold", color=VERDE).encode(
        x="exc_pct:Q", y="rot_pct:Q", text="modelo")

    st.altair_chart((curva + refs + etiq + aqui + etiq_aqui).properties(height=420),
                    width="stretch")
    st.caption("Curva azul: la frontera, un punto por cada nivel de cobertura. "
               "Cuadrados naranjas: tres formas de decidir el pedido que no eligen "
               "cuantil. Triángulo verde: donde te sitúan ahora mismo tus costes y "
               "tus ajustes.")

    st.divider()
    st.subheader("Qué demuestra este gráfico")

    ing = referencias[referencias.modelo == "Repetir la semana pasada"].iloc[0]
    gap = ing.rot_pct - float(np.interp(ing.exc_pct, fr.exc_pct, fr.rot_pct))
    izq, der = st.columns(2)
    with izq:
        st.markdown(
            "**Que la frontera existe y se puede recorrer.** Ese es el resultado "
            "que persigue el trabajo: no una cifra, sino el conjunto de "
            "compromisos alcanzables. Sin regresión cuantílica no hay curva, hay "
            "un punto suelto y ninguna forma fundamentada de moverse de él.")
        st.markdown(
            f"**Que repetir la semana pasada queda fuera.** Con el mismo excedente "
            f"que la política ingenua, la frontera se queda **{dec(gap)} puntos** "
            f"por debajo en rotura, unas {num(gap / 100 * dem)} unidades que sobran "
            "por no mirar más allá de la semana anterior.")
    with der:
        st.markdown(
            "**Lo que este gráfico no dice.** El XGBoost sin asimetría cae "
            "prácticamente sobre la frontera en su propio punto. Predecir bien la "
            "media no es ineficiente: lo que pasa es que te deja donde salga, sin "
            "manera de elegir otro sitio.")
        st.markdown(
            "**Dónde sí gana elegir el cuantil.** La alternativa clásica para "
            "moverse es multiplicar la predicción por un colchón de seguridad. "
            "Comparadas al mismo excedente, esa vía pierde 0,7 puntos de rotura "
            "con un colchón del −10 % y 1,2 con uno del −20 %. La ventaja aparece "
            "justo en el lado agresivo, que es donde se reduce el desperdicio.")

    st.divider()
    st.subheader("Por qué tu punto queda por encima de la curva")
    unico = cuantil_unico(c_sobra["Perecedero"], c_sobra["Aguanta"], c_falta)
    fila = unico.sort_values("coste").iloc[0]
    c_dos = d.coste_sin_ajustes.sum()
    ahorra = float(fila.coste) - c_dos
    st.markdown(
        "La curva está medida en **unidades**, y ahí tu punto queda por encima. "
        "No es un defecto: es que la curva aplica el mismo nivel a todo el "
        "surtido y tú aplicas uno al fresco y otro al resto. En unidades esa "
        "mezcla parece peor, porque una unidad de yogur cuenta igual que una de "
        "carne. En euros no lo es.")
    x, y, z = st.columns(3)
    x.metric(f"Mejor cuantil único (el {int(fila.Cuantil)} %)", eur(fila.coste))
    y.metric(f"Dos grupos ({optimo['Perecedero']} % y {optimo['Aguanta']} %)", eur(c_dos))
    z.metric("Diferencia", f"{signo(-ahorra)}{eur(abs(ahorra))}",
             delta=f"{dec(100 * ahorra / fila.coste)} % más barato" if ahorra > 0
                   else f"{dec(100 * abs(ahorra) / fila.coste)} % más caro",
             delta_color="normal" if ahorra > 0 else "inverse")
    st.caption("Separar por perecibilidad pierde en el plano de unidades y gana en "
               "el de dinero, que es donde se paga. Esa es la razón de que la "
               "herramienta no aplique un cuantil único, y de que el precio de "
               "cada categoría salga del propio conjunto de datos.")

    st.info(
        "Las cifras de esta pestaña no coinciden con las de la Figura 16 de la "
        "memoria. Allí la frontera se traza aplicando un colchón del 10 % sobre "
        "cada predicción; aquí no se aplica ninguno, porque sumar un colchón a un "
        "cuantil duplica el margen de seguridad: el cuantil ya codifica el nivel "
        "de servicio que se busca. La forma de la curva y las conclusiones son "
        "las mismas.", icon="ℹ️")

# ----------------------------------------------------- 4. por categoría
with t4:
    st.subheader("¿Prefieres que sobre o que falte?")
    st.markdown(
        "La cobertura de la izquierda sale de tus costes y se aplica en dos "
        "grupos. Pero tú sabes cosas que el histórico no dice: que el pan vacío "
        "a las siete de la tarde hace daño aunque salga barato, que en carne "
        "prefieres quedarte corto, que ese proveedor tiene pedido mínimo. "
        "**Baja el número para que falte antes que sobre; súbelo para lo "
        "contrario.** Lo que hace la herramienta es ponerle precio a esa decisión.")

    tabla = (d.drop_duplicates("COMMODITY_DESC")[["COMMODITY_DESC", "regimen", "nivel"]]
              .rename(columns={"regimen": "Tipo", "nivel": "Cobertura (%)"}))
    tabla["Categoría"] = tabla.COMMODITY_DESC.map(bonito)
    tabla = tabla.sort_values("Categoría").reset_index(drop=True)
    orden = tabla.COMMODITY_DESC.tolist()
    base_cat = dict(zip(d.drop_duplicates("COMMODITY_DESC").COMMODITY_DESC,
                        d.drop_duplicates("COMMODITY_DESC").nivel_base))

    ed = st.data_editor(
        tabla[["Categoría", "Tipo", "Cobertura (%)"]],
        width="stretch", hide_index=True, height=360,
        disabled=["Categoría", "Tipo"], num_rows="fixed", key="editor",
        column_config={"Cobertura (%)": st.column_config.SelectboxColumn(
            options=QS, required=True,
            help="Más bajo, más riesgo de que falte. Más alto, más riesgo de "
                 "que sobre.")})

    nuevos = {c: int(n) for c, n in zip(orden, ed["Cobertura (%)"])
              if int(n) != int(base_cat[c])}
    if nuevos != st.session_state.ajustes:
        st.session_state.ajustes = nuevos
        st.rerun()

    st.divider()
    if st.session_state.ajustes:
        n_aj = len(st.session_state.ajustes)
        dif_coste = coste_pedido - d.coste_sin_ajustes.sum()
        dif_exc = d.exc_pedido.sum() - d.exc_sin_ajustes.sum()
        dif_rot = d.rot_pedido.sum() - d.rot_sin_ajustes.sum()
        st.subheader(f"Lo que cuesta {'tu ajuste' if n_aj == 1 else f'tus {n_aj} ajustes'}")
        u, v, w = st.columns(3)
        u.metric("Lo que sobra", f"{signo(dif_exc)}{num(abs(dif_exc))} uds")
        v.metric("Lo que falta", f"{signo(dif_rot)}{num(abs(dif_rot))} uds")
        w.metric("Coste total", f"{signo(dif_coste)}{eur(abs(dif_coste))}")
        st.caption("Comparado con dejar cada categoría en el nivel de su grupo, "
                   "sumando las quince semanas. Signo positivo, más que antes; "
                   "negativo, menos. Que salga más caro no significa que te "
                   "equivoques: significa cuánto estás pagando por la razón que "
                   "tú tienes y el histórico no recoge. Esa es justo la decisión "
                   "que la herramienta no puede tomar.")
        st.warning(
            "Esto es lo que habría pasado en esas quince semanas concretas, no lo "
            "que va a pasar. Si vas probando ajustes hasta que el coste baje, "
            "estarás afinando sobre el pasado, que es precisamente el error que "
            "hace que la herramienta no ajuste sola por categoría.", icon="⚠️")

        def limpiar():
            st.session_state.ajustes = {}
            st.session_state.pop("editor", None)

        st.button("Quitar el ajuste" if n_aj == 1 else "Quitar todos los ajustes",
                  on_click=limpiar)
    else:
        st.subheader("Por qué esto no lo hace la herramienta sola")
        st.markdown(
            "Se probó. Ajustando el nivel de cada categoría con las primeras ocho "
            "semanas y cobrándolo en las siete siguientes, sale un **7,6 % más "
            "caro** que aplicar el nivel del grupo. El nivel que parece mejor en "
            "la primera mitad del periodo solo coincide con el de la segunda en "
            "**16 de las 157 categorías**, y la discrepancia mediana es de 15 "
            "puntos. Con quince semanas por categoría lo que se aprende es ruido. "
            "Por eso el ajuste está aquí, en tus manos, y no automatizado.")

# ------------------------------------------------------- 5. una categoría
with t5:
    cate = st.selectbox("Categoría", CATEGORIAS, format_func=bonito,
                        index=CATEGORIAS.index("BEEF") if "BEEF" in CATEGORIAS else 0)
    info = cat[cat.COMMODITY_DESC == cate].iloc[0]
    sc = d[d.COMMODITY_DESC == cate]
    j, k, l = st.columns(3)
    j.metric("Precio medio por unidad", eur(info.precio, 2))
    k.metric("Nivel que se le aplica", f"{int(sc.nivel.iloc[0])} %")
    l.metric("Le habría sobrado", f"{num(sc.exc_pedido.sum())} uds")

    hh = hist[hist.COMMODITY_DESC == cate][["WEEK_NO", "unidades"]].copy()
    hh["Serie"] = "Lo que se vendió"
    pp = sc[["WEEK_NO", "pedido"]].rename(columns={"pedido": "unidades"}).copy()
    pp["Serie"] = "Lo que habrías pedido"
    st.altair_chart(alt.Chart(pd.concat([hh, pp])).mark_line().encode(
        x=alt.X("WEEK_NO", title="Semana", scale=alt.Scale(zero=False)),
        y=alt.Y("unidades", title="Unidades"),
        color=alt.Color("Serie", scale=alt.Scale(
            domain=["Lo que se vendió", "Lo que habrías pedido"], range=[GRIS, VERDE]),
            legend=alt.Legend(orient="top", title=None)),
        tooltip=["WEEK_NO", "Serie", alt.Tooltip("unidades", format=".0f")]
    ).properties(height=340), width="stretch")
    st.caption("La línea gris recorre las 87 semanas de historia. La verde solo "
               "aparece en las quince finales, que el sistema no usó para aprender.")

st.divider()
st.caption("Trabajo Fin de Máster · Máster Universitario en Ciencia de Datos · "
           "Universidad de Alicante · Datos: Dunnhumby The Complete Journey. "
           "Los precios salen del propio conjunto de datos; los tres parámetros "
           "de coste los aporta el usuario, porque no constan en él.")
