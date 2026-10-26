#!/usr/bin/env python3
"""Cuaderno de los capítulos 6 y 8: RAG sobre documentos propios y evaluación."""
import json, sys
sys.path.insert(0, ".")
from _cabecera import portada, md, code, cuaderno

celdas = portada(
    "RAG: conectar la IA con tus documentos",
    "Capítulos 6 y 8",
    "Darle al modelo acceso a documentos que no vio al entrenarse: trocear, indexar, "
    "recuperar, responder citando las fuentes y **medir si la recuperación funciona**, "
    "que es el paso que casi nadie hace.",
    "Corre en CPU. Descarga dos modelos pequeños: uno de *embeddings* y uno de 1,5 B.")

celdas += [
md("""## Cómo leer este cuaderno

El libro trabaja con **seis PDF internos de una cooperativa eléctrica** que no se distribuyen.
Para que el cuaderno corra sin descargar nada, aquí se **generan tres PDF de juguete** con la
misma forma: condiciones, tarifas y guía de la factura.

Eso cambia una cosa importante y conviene decirla antes: **las cifras de este cuaderno son de
este corpus de juguete, no las del libro.** Lo que sí se reproduce es el procedimiento entero
—trocear, indexar, recuperar, responder y medir— y el orden de magnitud de lo que deberías
esperar con documentos de verdad."""),

md("""## 1. Instalación"""),
code("""%pip install -q "sentence-transformers==6.1.0" "transformers==5.19.0" \\
    "pypdf" "reportlab" "accelerate"
"""),

md("""## 2. Tres PDF de juguete, con la forma de los de verdad

No son los documentos del libro: son tres páginas escritas para que el buscador tenga algo
real que encontrar, con cifras concretas y vocabulario de factura eléctrica."""),
code('''import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

os.makedirs("documentos", exist_ok=True)
DOCS = {
 "condiciones.pdf": [
  "CONDICIONES GENERALES DEL CONTRATO DE SUMINISTRO",
  "1. El contrato se formaliza por un periodo inicial de doce meses y se prorroga",
  "   automaticamente salvo denuncia con quince dias de antelacion.",
  "2. La baja puede solicitarse en cualquier momento sin penalizacion a partir del",
  "   decimotercer mes de vigencia.",
  "3. El cambio de potencia contratada solo puede solicitarse una vez cada doce meses.",
 ],
 "tarifas.pdf": [
  "TARIFAS VIGENTES - EJERCICIO 2026",
  "Termino de potencia (periodo punta): 23,47 EUR por kW y ano.",
  "Termino de potencia (periodo valle): 1,42 EUR por kW y ano.",
  "Termino de energia (punta): 0,186 EUR por kWh.",
  "Termino de energia (llano): 0,142 EUR por kWh.",
  "Termino de energia (valle): 0,098 EUR por kWh.",
  "El impuesto electrico aplicable es del 3,8 por ciento.",
 ],
 "guia-factura.pdf": [
  "GUIA PARA ENTENDER TU FACTURA",
  "El importe total se compone de tres bloques: potencia, energia e impuestos.",
  "El termino de potencia se paga siempre, aunque el consumo del mes sea cero.",
  "El termino de energia depende de los kilovatios hora consumidos en cada periodo.",
  "Si detectas un cobro duplicado, dispones de seis anos para reclamarlo.",
 ],
}
for nombre, lineas in DOCS.items():
    c = canvas.Canvas(f"documentos/{nombre}", pagesize=A4)
    y = 780
    for i, l in enumerate(lineas):
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 13 if i == 0 else 10)
        c.drawString(60, y, l); y -= 22
    c.save()
print("PDF creados:", sorted(os.listdir("documentos")))
'''),

md("""## 3. Trocear: el paso que decide la calidad de todo lo demás

Un documento entero no cabe en el contexto y, aunque cupiera, el buscador funciona peor con
textos largos. Se parte en **trozos con solape**: el solape evita que una frase partida por la
mitad se pierda en los dos lados.

Y se guarda **de dónde viene cada trozo** — documento y página—. Sin eso no se pueden citar las
fuentes, y un asistente que no cita no se puede comprobar."""),
code('''import glob
from pypdf import PdfReader

def trocear(texto, tam=800, solape=150):
    trozos, i = [], 0
    while True:
        trozos.append(texto[i:i + tam])
        if i + tam >= len(texto):      # el último trozo ya llega al final: no crear uno redundante
            break
        i += tam - solape
    return trozos

fragmentos, metadatos = [], []
for ruta in sorted(glob.glob("documentos/*.pdf")):
    for n_pag, pagina in enumerate(PdfReader(ruta).pages, start=1):
        for trozo in trocear(pagina.extract_text()):
            fragmentos.append(trozo)
            metadatos.append({"documento": os.path.basename(ruta), "pagina": n_pag})
print(len(fragmentos), "fragmentos ·", metadatos[0])
print(repr(fragmentos[1][:120]))
'''),
md("""**Qué deberías ver.** Tres fragmentos —uno por PDF, porque son páginas cortas— con su
documento y su página. Con los seis PDF del libro salen bastantes más.

**Y fíjate en el texto extraído**: el PDF no devuelve lo que ves, devuelve lo que el extractor
consigue reconstruir. Con documentos escaneados, esto devuelve basura o nada, y ahí es donde
empieza el trabajo de verdad de un RAG."""),

md("""## 4. Indexar y buscar

Cada fragmento se convierte en un vector, y buscar es comparar el vector de la pregunta con los
de los fragmentos. **El prefijo `passage:` y `query:` no es decoración**: este modelo se
entrenó con ellos y sin ellos recupera peor.

El libro usa `intfloat/multilingual-e5-base` por una razón medida: con otros modelos
multilingües la similitud se quedaba por debajo del umbral y el buscador elegía el fragmento
equivocado."""),
code('''from sentence_transformers import SentenceTransformer
import numpy as np

emb = SentenceTransformer("intfloat/multilingual-e5-base")
vectores = emb.encode(["passage: " + f for f in fragmentos], normalize_embeddings=True)

def buscar(pregunta, k=3):
    q = emb.encode(["query: " + pregunta], normalize_embeddings=True)[0]
    puntuaciones = vectores @ q           # similitud coseno (vectores normalizados)
    mejores = np.argsort(-puntuaciones)[:k]
    return [(fragmentos[i], metadatos[i], float(puntuaciones[i])) for i in mejores]

for texto, meta, punt in buscar("¿Cuánto cuesta el kW de potencia en el periodo valle?"):
    print(f"{punt:.3f}  {meta['documento']} p.{meta['pagina']}  ·  {texto[:70]}…")
'''),
md("""**Qué deberías ver.** `tarifas.pdf` el primero, con una puntuación claramente por encima
de los otros dos. **Mira la distancia entre el primero y el segundo**, no solo quién gana: si
están pegados, el buscador no está seguro, y eso es una señal de que el troceado o el modelo no
sirven para tus documentos."""),

md("""## 5. Responder citando las fuentes

El modelo recibe los fragmentos y la pregunta, con una instrucción que hace dos cosas:
**obligarle a citar** y **permitirle decir que no lo sabe**. La segunda es la que evita que se
invente la respuesta cuando el contexto no la tiene."""),
code('''from transformers import pipeline

chat = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")

def responder(pregunta, k=3):
    recuperados = buscar(pregunta, k)
    contexto = "\\n\\n".join(f"[{i+1}] {t}" for i, (t, _, _) in enumerate(recuperados))
    mensajes = [
        {"role": "system", "content": "Responde solo con la información del contexto. "
         "Cita las fuentes como [n]. Si no está en el contexto, di que no lo sabes."},
        {"role": "user", "content": f"Contexto:\\n{contexto}\\n\\nPregunta: {pregunta}"},
    ]
    salida = chat(mensajes, max_new_tokens=200, do_sample=False)
    return salida[0]["generated_text"][-1]["content"], recuperados

texto, fuentes = responder("¿Cuánto cuesta el kW de potencia contratada en el periodo valle?")
print(texto)
print("\\nfuentes:", [(m["documento"], m["pagina"]) for _, m, _ in fuentes])
'''),
md("""**Qué deberías ver.** Una respuesta con **1,42 euros por kW y año** y una cita `[1]`.

**La prueba de verdad es la siguiente celda**, no esta: un RAG que contesta bien lo que está en
los documentos es fácil; lo difícil es que **no conteste** lo que no está."""),
code('''texto, _ = responder("¿Cuál es el horario de atención telefónica?")
print(texto)
'''),
md("""**Qué deberías ver.** Que dice que no lo sabe, o que no está en el contexto. **Ninguno de
los tres documentos menciona un horario**, así que cualquier horario concreto que aparezca ahí
es inventado.

Esta es la pregunta que hay que hacerle a cualquier RAG antes de enseñárselo a nadie: una cuya
respuesta **no esté** en los documentos."""),

md("""# Capítulo 8 · Evaluar

## 6. Recall@k: ¿llega el fragmento correcto?

Antes de evaluar la respuesta hay que evaluar la **recuperación**, porque si el fragmento
bueno no llega, el modelo no puede acertar por mucho que mejore el prompt.

`Recall@k` mide en qué proporción de las preguntas el fragmento con la evidencia está entre los
`k` recuperados. Se necesita un conjunto de preguntas con su evidencia anotada a mano: aquí
son cinco, en el libro son ochenta."""),
code('''conjunto_eval = [
    {"pregunta": "¿Cuánto cuesta el kW de potencia en valle?",      "evidencia": "1,42 EUR por kW"},
    {"pregunta": "¿Qué impuesto se aplica a la electricidad?",      "evidencia": "3,8 por ciento"},
    {"pregunta": "¿Cuánto tiempo tengo para reclamar un cobro duplicado?", "evidencia": "seis anos"},
    {"pregunta": "¿Puedo darme de baja sin penalización?",          "evidencia": "sin penalizacion"},
    {"pregunta": "¿Cada cuánto puedo cambiar la potencia?",         "evidencia": "una vez cada doce meses"},
]

for k in (1, 2, 3):
    aciertos = sum(any(c["evidencia"] in t for t, _, _ in buscar(c["pregunta"], k))
                   for c in conjunto_eval)
    print(f"Recall@{k}: {aciertos / len(conjunto_eval):.0%}  ({aciertos}/{len(conjunto_eval)})")
'''),
md("""**Qué deberías ver: 100 % en los tres, y eso es exactamente el problema.** Medido al
montar este cuaderno: Recall@1, @2 y @3 los tres al 100 %.

**Ese 100 % no mide la calidad de nada.** Hay tres fragmentos, uno por documento, y las cinco
preguntas tienen su evidencia en documentos distintos: acertar el primero es casi inevitable.
Un Recall@1 del 100 % con tres candidatos y uno correcto es lo que saldría tirando una moneda
con tres caras y dos trucadas.

**Esa es la trampa de medir con un corpus de juguete**: el número sale bonito por el tamaño, no
por la calidad, y sale bonito igual si el buscador es malo. Con los seis PDF del libro y
ochenta preguntas, el mismo número sí dice algo. Lo que este cuaderno te enseña es el
procedimiento; **el número hay que sacarlo de tus documentos**, y si te sale 100 % a la primera,
sospecha del conjunto de evaluación antes de celebrarlo."""),
code('''# Dónde falla: la pregunta cuyo fragmento NO llega el primero
for c in conjunto_eval:
    rec = buscar(c["pregunta"], 3)
    pos = next((i for i, (t, _, _) in enumerate(rec) if c["evidencia"] in t), None)
    print(f"{'ok ' if pos == 0 else 'POS ' + str(pos):6} {c['pregunta'][:52]:54} {rec[0][2]:.3f}")
'''),
md("""**Qué deberías ver.** Cinco `ok` y las puntuaciones entre 0,81 y 0,88.

**Esta tabla es la que dice qué arreglar** cuando el Recall no sale al 100 %: una evidencia que
llega en posición 1 o 2 se arregla subiendo `k`; una que no llega es un problema de troceado o
de modelo de *embeddings*. Y la columna de puntuación avisa de otra cosa: **si el primero gana
por poco, el buscador no está seguro** aunque acierte.

Aquí todas pasan de 0,81, que con `multilingual-e5-base` es una separación holgada. Si con tus
documentos te salen puntuaciones en torno a 0,5 aunque el fragmento sea el bueno, el problema
no es el modelo de lenguaje: es el troceado."""),
]

json.dump(cuaderno(celdas), open("rag-conectar-la-ia-con-tus-documentos.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("escrito rag-conectar-la-ia-con-tus-documentos.ipynb")
