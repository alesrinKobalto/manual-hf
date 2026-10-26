#!/usr/bin/env python3
"""Validador del dataset de la cooperativa: la segunda de las cuatro capas del anexo B.

Comprueba, una por una, las afirmaciones que el libro hace sobre este dataset. No
comprueba «que esté bien» en abstracto: comprueba cifras publicadas, y falla con
código 1 si alguna deja de cumplirse.

    python herramientas/valida_dataset.py

Lo que verifica y dónde lo dice el libro:

* 6 PDF, y su texto extraído coincide con el fuente de `documentos/fuente/` (cap. 11).
* 43 fragmentos troceando página a página y 41 troceando el documento completo,
  con el `trocear()` del capítulo 6 (apartados 6.2 y 8.3).
* El primer fragmento es de `ayudas_socios.pdf`, página 1 (apartado 6.2).
* 60 preguntas factuales y 60 evidencias que aparecen **literalmente dentro de un
  fragmento**, no solo en el PDF: si una evidencia se partiera entre dos fragmentos,
  el Recall@k del capítulo 8 no podría encontrarla nunca (anexo A: «60/60 evidencias»).
* 80 preguntas de evaluación, cinco de tipo `inyeccion` (apartado 19.3).
* 500 conversaciones y 500 distintas con el umbral 0,6 del capítulo 16
  (anexo A: «500/500 conversaciones distintas»).
* 0 preguntas de evaluación contaminadas con el umbral 0,5 (anexo A y ejercicio 3
  del capítulo 16).
* Ausencia de datos personales: `pasa_filtros` del capítulo 16 sobre las 500
  respuestas, más una pasada de DNI, IBAN y teléfono sobre el dataset entero.

Las funciones de normalización, deduplicación y contaminación son las del apartado
16.2, copiadas tal cual: si el libro las cambia, hay que cambiarlas aquí.
"""
import glob
import json
import os
import re
import sys
import unicodedata

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS = os.path.join(RAIZ, "datos", "cooperativa")
DOCS = os.path.join(DATOS, "documentos")

ESPERADO = {
    "documentos": 6,
    "fragmentos_pagina": 43,
    "fragmentos_documento": 41,
    "primer_documento": "ayudas_socios.pdf",
    "factuales": 60,
    "evaluacion": 80,
    "inyeccion": 5,
    "conversaciones": 500,
}

fallos, avisos = [], []


def comprueba(condicion, mensaje):
    print(("  OK   " if condicion else "  FALLA") + "  " + mensaje)
    if not condicion:
        fallos.append(mensaje)


# ── funciones del apartado 16.2, literales ───────────────────────────────────
def normalizar(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9ñ ]+", " ", t).split()


def ngramas(t, n=3):
    p = normalizar(t)
    return {tuple(p[i:i + n]) for i in range(max(len(p) - n + 1, 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def deduplicar(textos, umbral=0.6):
    conservados, firmas = [], []
    for t in textos:
        f = ngramas(t)
        if all(jaccard(f, g) < umbral for g in firmas):
            conservados.append(t)
            firmas.append(f)
    return conservados


def contaminados(entrenamiento, evaluacion, umbral=0.5):
    firmas = [ngramas(t) for t in entrenamiento]
    return [q for q in evaluacion if any(jaccard(ngramas(q), f) >= umbral for f in firmas)]


def pasa_filtros(ej):
    r = ej["respuesta"]
    if not 20 <= len(r) <= 1200:
        return False, "longitud"
    if re.search(r"\b\d{8}[A-HJ-NP-TV-Z]\b", r):
        return False, "posible DNI"
    if re.search(r"(?i)como modelo de lenguaje|as an ai", r):
        return False, "muletilla de modelo"
    return True, "ok"


# ── troceado del capítulo 6, literal ────────────────────────────────────────
def trocear(texto, tam=800, solape=150):
    trozos, i = [], 0
    while True:
        trozos.append(texto[i:i + tam])
        if i + tam >= len(texto):
            break
        i += tam - solape
    return trozos


def main():
    from pypdf import PdfReader

    print("\n1. DOCUMENTOS")
    pdfs = sorted(glob.glob(os.path.join(DOCS, "*.pdf")))
    comprueba(len(pdfs) == ESPERADO["documentos"],
              f"{len(pdfs)} PDF (esperados {ESPERADO['documentos']})")
    comprueba(os.path.basename(pdfs[0]) == ESPERADO["primer_documento"],
              f"el primero por orden alfabético es {os.path.basename(pdfs[0])}")

    paginas_de = {}
    for ruta in pdfs:
        paginas_de[os.path.basename(ruta)] = [p.extract_text() for p in PdfReader(ruta).pages]

    # el PDF es un derivado: lo que manda es el texto fuente
    for nombre, paginas in paginas_de.items():
        fuente = os.path.join(DOCS, "fuente", nombre.replace(".pdf", ".txt"))
        if not os.path.exists(fuente):
            comprueba(False, f"falta el fuente de {nombre}")
            continue
        with open(fuente, encoding="utf-8") as f:
            esperadas = [p.strip("\n") for p in f.read().split("---PAGINA---")]
        iguales = len(esperadas) == len(paginas) and all(
            [l for l in e.split("\n") if l.strip()] == [l for l in p.split("\n") if l.strip()]
            for e, p in zip(esperadas, paginas))
        comprueba(iguales, f"{nombre}: el texto extraído coincide con su fuente")

    print("\n2. TROCEADO (capítulo 6)")
    frag_pag, meta = [], []
    for nombre, paginas in paginas_de.items():
        for n_pag, texto in enumerate(paginas, start=1):
            for trozo in trocear(texto):
                frag_pag.append(trozo)
                meta.append({"documento": nombre, "pagina": n_pag})
    frag_doc = [t for paginas in paginas_de.values() for t in trocear("\n".join(paginas))]
    comprueba(len(frag_pag) == ESPERADO["fragmentos_pagina"],
              f"{len(frag_pag)} fragmentos página a página (esperados {ESPERADO['fragmentos_pagina']})")
    comprueba(len(frag_doc) == ESPERADO["fragmentos_documento"],
              f"{len(frag_doc)} fragmentos por documento completo (esperados {ESPERADO['fragmentos_documento']})")
    comprueba(meta[0] == {"documento": ESPERADO["primer_documento"], "pagina": 1},
              f"el primer fragmento es {meta[0]}")

    print("\n3. EVALUACIÓN")
    with open(os.path.join(DATOS, "eval.jsonl"), encoding="utf-8") as f:
        ev = [json.loads(l) for l in f if l.strip()]
    comprueba(len(ev) == ESPERADO["evaluacion"],
              f"{len(ev)} preguntas de evaluación (esperadas {ESPERADO['evaluacion']})")
    factuales = [c for c in ev if c["tipo"] == "factual"]
    inyeccion = [c for c in ev if c["tipo"] == "inyeccion"]
    comprueba(len(factuales) == ESPERADO["factuales"], f"{len(factuales)} de tipo factual")
    comprueba(len(inyeccion) == ESPERADO["inyeccion"], f"{len(inyeccion)} de tipo inyeccion")
    comprueba(len({c["id"] for c in ev}) == len(ev), "los identificadores no se repiten")

    con_evidencia = 0
    for c in factuales:
        dentro = [i for i, t in enumerate(frag_pag) if c["evidencia"] in t]
        if dentro:
            con_evidencia += 1
            if meta[dentro[0]]["documento"] != c["documento"]:
                avisos.append(f"{c['id']}: la evidencia aparece antes en "
                              f"{meta[dentro[0]]['documento']} que en {c['documento']}")
        else:
            fallos.append(f"{c['id']}: la evidencia no cabe entera en ningún fragmento")
    comprueba(con_evidencia == len(factuales),
              f"{con_evidencia}/{len(factuales)} evidencias dentro de un fragmento")

    print("\n4. ENTRENAMIENTO")
    with open(os.path.join(DATOS, "sft.jsonl"), encoding="utf-8") as f:
        sft = [json.loads(l) for l in f if l.strip()]
    comprueba(len(sft) == ESPERADO["conversaciones"],
              f"{len(sft)} conversaciones (esperadas {ESPERADO['conversaciones']})")
    formas = {tuple(m["role"] for m in c["messages"]) for c in sft}
    comprueba(formas == {("system", "user", "assistant")},
              f"todas con la forma system/user/assistant ({len(formas)} forma(s) distinta(s))")
    preguntas = [c["messages"][1]["content"] for c in sft]
    respuestas = [c["messages"][2]["content"] for c in sft]
    distintas = deduplicar(preguntas)
    comprueba(len(distintas) == len(preguntas),
              f"{len(distintas)}/{len(preguntas)} conversaciones distintas (umbral 0,6)")

    print("\n5. CONTAMINACIÓN")
    sucias = contaminados(preguntas, [c["pregunta"] for c in ev])
    comprueba(not sucias, f"{len(sucias)} preguntas de evaluación contaminadas (umbral 0,5)")
    for q in sucias:
        print(f"         · «{q}»")

    print("\n6. DATOS PERSONALES Y FILTROS BARATOS")
    malas = [(p, pasa_filtros({"respuesta": r})[1])
             for p, r in zip(preguntas, respuestas) if not pasa_filtros({"respuesta": r})[0]]
    comprueba(not malas, f"{len(malas)} respuestas no pasan pasa_filtros()")
    for p, motivo in malas[:5]:
        print(f"         · {motivo}: «{p[:60]}»")

    todo = "\n".join(preguntas + respuestas
                     + [c.get("pregunta", "") + " " + c.get("respuesta", "") for c in ev]
                     + [t for paginas in paginas_de.values() for t in paginas])
    patrones = {
        "DNI": r"\b\d{8}[A-HJ-NP-TV-Z]\b",
        "IBAN español": r"\bES\d{2}[ ]?\d{4}",
        "teléfono": r"\b[6789]\d{2}[ .-]?\d{3}[ .-]?\d{3}\b",
        "correo": r"[\w.+-]+@[\w-]+\.[\w.]+",
    }
    for nombre, patron in patrones.items():
        halla = sorted(set(re.findall(patron, todo)))
        comprueba(not halla, f"sin {nombre} en el dataset"
                  + (f" — encontrado: {halla[:3]}" if halla else ""))

    print()
    if avisos:
        print(f"{len(avisos)} aviso(s) que no hacen fallar la validación:")
        for a in avisos:
            print("  ·", a)
        print()
    if fallos:
        print(f"VALIDACIÓN FALLIDA: {len(fallos)} comprobación(es) incumplida(s)")
        for f_ in fallos:
            print("  ·", f_)
        return 1
    print("Validación superada: el dataset cumple todo lo que el libro afirma de él.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
