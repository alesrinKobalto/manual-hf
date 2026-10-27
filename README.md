# manual-hf · código, dataset y erratas de «Crea tu propia IA»

Repositorio del libro **Crea tu propia IA con modelos de código abierto de Hugging
Face**, de Alejandra Espinosa (Kobalto Educación, 2026). El libro está en
[kobalto.es/biblioteca](https://kobalto.es/biblioteca/).

Aquí hay tres cosas: el **dataset de la cooperativa** que recorre el libro entero, el
**código del proyecto final** y la puerta para **avisar de una errata**.

## Empezar en dos minutos, sin GPU y sin descargar modelos

```bash
pip install -r requirements-datos.txt
python herramientas/valida_dataset.py        # el dataset, comprobado cifra por cifra
python datos/cooperativa/linea_base_bm25.py  # la línea base del recuperador
pytest -q
```

Tres paquetes, un par de segundos y ningún modelo. Es a propósito: lo que vigila que el
libro siga siendo cierto tiene que poder ejecutarlo cualquiera.

## Estructura

La del apartado 11.1 del libro:

```
manual-hf/
├── datos/cooperativa/       # el dataset: ver su propio README
│   ├── documentos/          # 6 PDF ficticios y su texto fuente
│   ├── sft.jsonl            # 500 conversaciones para el ajuste fino
│   ├── eval.jsonl           # 80 preguntas de evaluación (nunca se entrena con ellas)
│   └── linea_base_bm25.py   # la cifra que hay que superar, sin modelos
├── indexar.py               # capítulo 6: trocear y calcular embeddings
├── entrenar.py              # capítulo 7: LoRA (y QLoRA) con TRL
├── evaluar.py               # capítulo 8: Recall@k y respuestas del modelo
├── app.py                   # capítulo 9: interfaz Gradio
├── recuperador.py           # la búsqueda, compartida por evaluar.py y app.py
├── cuadernos/               # los seis cuadernos de Kaggle y sus fuentes
├── herramientas/            # validador del dataset, cuadernos, vigilancia de versiones
├── docs/lectores_beta.md    # el protocolo de la tercera capa de validación
└── licencias/               # los tres textos legales completos
```

`recuperador.py` no está en el esquema del libro: se ha sacado aparte porque `evaluar.py`
y `app.py` tienen que buscar **exactamente igual**. Si fueran dos funciones distintas,
el Recall@4 medido no diría nada de lo que ve el usuario.

## El proyecto completo

```bash
pip install -r requirements.txt
python indexar.py                     # 43 fragmentos -> índice de embeddings
python evaluar.py                     # Recall@1, @4 y @10 sobre las 60 factuales
python entrenar.py                    # adaptador LoRA (necesita GPU)
python evaluar.py --responder --adaptador asistente-cooperativa-lora
python app.py                         # la demo en local
```

El orden importa y es el del libro: **primero la línea base, después las mejoras**. Sin
la cifra de la fase 1 ninguna de las siguientes puede demostrar que aporta algo
(apartado 11.2).

## Los cuadernos

Están en **Kaggle** y se abren sin instalar nada: la tabla con los seis enlaces y sus
fuentes está en [`cuadernos/README.md`](cuadernos/README.md). Los cinco que tienen
fuente no se editan a mano, se regeneran:

```bash
python herramientas/generar_cuadernos.py
```

## Qué está comprobado y qué no

El libro no imprime ninguna salida que no se haya obtenido ejecutando el código, y este
repositorio sigue la misma regla. Lo que no se ha podido ejecutar, se dice:

| Qué | Cómo se comprueba | Estado |
|---|---|---|
| Dataset: documentos, troceado, evidencias, duplicados, contaminación, datos personales | `herramientas/valida_dataset.py`, en cada cambio | Ejecutado: pasa entero |
| Línea base del recuperador | `datos/cooperativa/linea_base_bm25.py` | Ejecutado: las cifras de las pruebas salen de ahí |
| Generadores del dataset y de los cuadernos | `pytest` compara byte a byte | Ejecutado: regeneran idéntico |
| Versiones fijadas | `herramientas/vigila_versiones.py`, cada lunes | Ejecutado contra PyPI |
| Las tres mejoras del apartado 6.5 (embeddings, híbrido, reranking) | `datos/cooperativa/mide_recuperador.py`, en CPU | Ejecutado: las dos tablas de la ficha del dataset, con las versiones fijadas y repetidas con otras |
| `indexar.py`, `entrenar.py`, `evaluar.py --responder`, `app.py` | Exigen descargar modelos, y `entrenar.py` una GPU | **No ejecutados en la integración continua**; el código es el de los capítulos 6, 7, 8 y 9, ejecutado al escribirlos |

Esa última fila es la razón de que los cuadernos de Kaggle existan: ahí ese código sí
corre, con GPU gratuita y sin instalar nada.

## Cómo contribuir

* **Erratas:** abra una incidencia con la plantilla **Errata**, indicando capítulo,
  texto actual, problema y propuesta. Los cuatro campos, porque con los cuatro una
  errata se resuelve sin preguntar nada más.
* **Código:** toda propuesta debe pasar `pytest` y, si cambia el libro, regenerar los
  cuadernos con `herramientas/generar_cuadernos.py`.
* **Explicaciones que no se entienden:** plantilla **Explicación confusa**. No es una
  errata y no se trata igual: alimenta la capa de lectores beta
  ([`docs/lectores_beta.md`](docs/lectores_beta.md)).
* **Estilo:** español de España, ejemplos del mundo real en lugar de analogías,
  versiones fijadas y salidas reales.

Si una prueba falla después de un cambio, el número de la prueba **no se toca para que
pase**: hay que decidir si lo que cambia es el dataset o lo que dice el libro, y dejar
las dos cosas de acuerdo. Es toda la utilidad que tiene este repositorio.

## Las cuatro capas de validación

Las del anexo B del libro. Dos viven aquí:

| Capa | Dónde | Cuándo |
|---|---|---|
| Pruebas automáticas | `.github/workflows/pruebas.yml` | En cada cambio y cada lunes |
| Validador del dataset | `herramientas/valida_dataset.py` | En cada cambio del dataset |
| Lectores beta | `docs/lectores_beta.md` | Antes de cada edición |
| Revisión experta | — | Antes de cada edición |

## Licencias

Código **Apache 2.0**, dataset **CC BY 4.0**, textos **CC BY-SA 4.0**. El reparto fichero
a fichero y los tres textos legales completos están en [`LICENSE.md`](LICENSE.md).

> Basado en *Crea tu propia IA con modelos de código abierto de Hugging Face*,
> de Alejandra Espinosa (Kobalto Educación, 2026) — https://kobalto.es/biblioteca/
