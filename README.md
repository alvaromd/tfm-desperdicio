# Reducción del desperdicio alimentario en cadenas de supermercados

Código y cuadro de mando del Trabajo Fin de Máster del Máster Universitario en
Ciencia de Datos de la Universidad de Alicante.

El trabajo aborda la reposición semanal en el comercio minorista de alimentación
como un problema de decisión bajo costes asimétricos: que sobre producto no
cuesta lo mismo que quedarse corto, de modo que la cantidad que conviene pedir no
es la demanda esperada sino un cuantil de su distribución. En lugar de buscar un
único punto óptimo, se cuantifica la frontera completa entre excedente y rotura y
se deja la elección del punto de operación en manos de quien conoce sus costes.

## Cuadro de mando

**[Ver la aplicación](https://tfm-desperdicio-hsmyazhyx4nggrylhszyn6.streamlit.app/)**

Convierte las predicciones de cuantil en una cantidad concreta a pedir de cada
categoría. No pregunta por el cuantil objetivo, que es jerga, sino por tres
porcentajes del negocio, y de ahí deduce el nivel de cobertura. Cada categoría se
valora a su precio real y el producto perecedero recibe un punto de operación
distinto del que aguanta, porque el sobrante de una lata se vende la semana
siguiente y el de un filete se tira.

Para ejecutarlo en local:

```bash
pip install -r requirements.txt
streamlit run cuadro_mando/app.py
```

## Contenido

| Ruta | Qué es |
| --- | --- |
| `cuadro_mando/app.py` | La aplicación Streamlit |
| `cuadro_mando/datos/` | Predicciones ya calculadas que consume la aplicación |
| `src/` | Los guiones de análisis, modelado y generación de figuras |
| `src/requirements-modelado.txt` | Dependencias para reproducir el modelado completo |

Los guiones más relevantes:

- `src/modelado.py`: comparativa de modelos, frontera excedente-rotura
- `src/red_neuronal.py`: perceptrón multicapa con pérdida cuantílica
- `src/redes_avanzadas.py`: red multicuantil, LSTM y embeddings de categoría
- `src/datos_cuadro_mando.py`: entrena los diecinueve cuantiles que lee la aplicación
- `src/referencias_frontera.py`: políticas de referencia del plano excedente-rotura

## Datos

El trabajo utiliza **The Complete Journey** de Dunnhumby, un conjunto de datos de
transacciones reales de una cadena de alimentación durante dos años. Los datos
originales **no se incluyen en este repositorio** y deben descargarse de la
fuente original.

Los ficheros de `cuadro_mando/datos/` no son datos originales: son predicciones
agregadas por categoría y semana, calculadas a partir de ellos.

Para reproducir el trabajo completo hay que colocar los CSV originales en
`data/raw/` y ejecutar los guiones de `src/` en este orden:

1. `eda.py` y `eda_hallazgos.py`: depuran los datos y construyen el panel de modelado
2. `modelado.py`: líneas base, XGBoost, SVR y la frontera excedente-rotura
3. `red_neuronal.py` y `redes_avanzadas.py`: perceptrón, red multicuantil, LSTM y embeddings
4. `segmentacion.py`, `reparto_topdown.py` y `comprobacion_muertos.py`: análisis complementarios
5. `datos_cuadro_mando.py` y `referencias_frontera.py`: ficheros que consume la aplicación
6. `figuras_memoria.py`, `figuras_comparativa.py`, `figuras_conceptuales.py` y `figuras_redes.py`: figuras de la memoria

## Reproducibilidad

Todos los modelos usan semilla fija y una partición temporal, nunca aleatoria: se
entrena con las semanas anteriores al corte y se evalúa sobre las quince finales,
que ningún modelo ve durante el entrenamiento. El cuadro de mando no entrena
nada, únicamente lee las predicciones ya calculadas.
