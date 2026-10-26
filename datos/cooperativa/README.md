# Dataset de la Cooperativa Eléctrica Valle Verde

El caso que recorre el libro entero: el servicio de atención a las personas socias de
una cooperativa eléctrica. Seis documentos internos, 500 conversaciones para el ajuste
fino y 80 preguntas de evaluación.

**La cooperativa no existe.** Ni sus precios, ni sus plazos, ni sus condiciones
corresponden a una entidad real ni a la normativa vigente. Están escritos para que un
sistema RAG tenga algo concreto y verificable que recuperar, y para que las respuestas
se puedan comprobar frente a una fuente.

Licencia: **CC BY 4.0** (los generadores, que son código, van bajo Apache 2.0). Ver
`LICENSE.md`.

## Qué hay

| Fichero | Qué es |
|---|---|
| `documentos/*.pdf` | Los seis documentos de la cooperativa, 12 páginas en total |
| `documentos/fuente/*.txt` | Su texto, que es lo que de verdad se versiona |
| `genera_pdf.py` | Reconstruye los PDF desde el texto, byte a byte iguales |
| `semillas.json` | 50 intenciones de usuario con sus formulaciones y respuestas |
| `genera_sft.py` | Empareja las semillas y escribe `sft.jsonl` |
| `sft.jsonl` | 500 conversaciones `system`/`user`/`assistant` |
| `eval.jsonl` | 80 preguntas de evaluación, que **nunca** se usan para entrenar |
| `linea_base_bm25.py` | La línea base del recuperador, sin modelos y sin GPU |

Los PDF se versionan aunque sean binarios, porque son lo que el libro promete y lo que
el troceado del capítulo 6 mide. Pero el original es el `.txt`: si hay que cambiar una
coma, se cambia ahí y se regenera.

## De dónde sale

Las 500 formulaciones y las 150 respuestas de `semillas.json` están **escritas a mano**:
ninguna la ha generado un modelo. El capítulo 16 describe el proceso industrial —un
experto escribe de 30 a 50 semillas y un modelo grande genera variaciones— y aquí se ha
hecho la parte de las semillas y se ha ampliado a mano, para que el dataset del libro no
dependa de la salida de ningún modelo ni de su licencia.

`genera_sft.py` solo empareja, sin azar: la formulación n de cada intención va con su
respuesta n módulo el número de respuestas. Dos ejecuciones dan el mismo fichero.

## Qué se le comprueba

    python herramientas/valida_dataset.py

| Comprobación | Resultado |
|---|---|
| Documentos | 6 PDF, y su texto extraído coincide con el fuente |
| Troceado página a página (cap. 6) | 43 fragmentos; el primero, `ayudas_socios.pdf` p. 1 |
| Troceado por documento completo | 41 fragmentos |
| Evaluación | 80 preguntas: 60 factuales, 5 de inyección, 7 sin respuesta, 4 ambiguas, 4 fuera de ámbito |
| Evidencias | 60/60 aparecen literalmente **dentro de un fragmento** |
| Entrenamiento | 500 conversaciones, 500 distintas con el umbral 0,6 del cap. 16 |
| Contaminación | 0 preguntas de evaluación con similitud ≥ 0,5 frente al entrenamiento |
| Datos personales | sin DNI, IBAN, teléfonos ni correos; `pasa_filtros()` en las 500 respuestas |

Que las evidencias estén **dentro de un fragmento** y no solo dentro del PDF es la
comprobación que de verdad importa: una evidencia partida entre dos fragmentos haría
que el Recall@k del capítulo 8 no pudiera encontrarla nunca, y la culpa parecería del
modelo de embeddings.

## La contaminación no fue hipotética

La primera versión de `sft.jsonl` tenía **37 preguntas de evaluación contaminadas**:
al escribir las formulaciones de entrenamiento se reutilizó la redacción de las
preguntas de evaluación, y veintiséis de ellas eran idénticas salvo un punto en lugar
de una coma —similitud 1,0 tras normalizar—. El validador las cazó todas y se
reescribió el lado del entrenamiento, nunca el de la evaluación.

Es el ejercicio 3 del capítulo 16 ocurriendo de verdad: sin esa comprobación, el
modelo ajustado habría sacado en esas 37 preguntas una nota que no refleja su
capacidad, y la mejora medida frente a la línea base habría sido en buena parte
memoria.

## Lo que este dataset no es

* **No es tráfico real.** Nadie ha escrito estas preguntas a un servicio de atención:
  están escritas imitando cómo se preguntan esas cosas. Un dataset real tiene faltas,
  mensajes a medias, personas enfadadas y preguntas que mezclan tres asuntos.
* **No mide calidad de respuesta.** `eval.jsonl` trae la respuesta de referencia y la
  evidencia, que sirven para el Recall@k y para el juez del capítulo 8; la decisión
  final sigue siendo humana, como dice el apartado 11.2.
* **No cubre las lenguas cooficiales.** Todo está en castellano. El capítulo 13
  explica por qué eso es una limitación y no un detalle.
