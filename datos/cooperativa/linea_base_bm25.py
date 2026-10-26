#!/usr/bin/env python3
"""Línea base del recuperador: BM25 sobre los seis PDF, sin modelos ni GPU.

Es la cifra de referencia del apartado 8.3 del libro: lo que los embeddings, la
búsqueda híbrida y el reranking del apartado 6.5 tienen que superar. Se ejecuta en
dos segundos y no descarga nada, así que sirve de prueba de humo del dataset.

    python datos/cooperativa/linea_base_bm25.py

Decisiones del buscador, explícitas porque cambian el resultado:

* BM25 de Okapi con k1 = 1,5 y b = 0,75, los valores habituales.
* **Sin lista de palabras vacías**: «de», «la» o «qué» cuentan como cualquier otra
  palabra. Quitarlas sube el recall; se deja a propósito para que la línea base sea
  la más tonta posible.
* Las cifras con coma decimal se mantienen como un solo término (`0,0105`), porque
  partirlas en `0` y `0105` destruye justo las preguntas de precio.
* Se trocea **página a página** con el `trocear()` del capítulo 6 (43 fragmentos).
  Con `--documento-completo` se trocea el texto corrido de cada PDF (41 fragmentos),
  que es la comparación del libro: ninguna de las dos formas gana en todos los k.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
PDFS = os.path.join(AQUI, "documentos", "*.pdf")
EVAL = os.path.join(AQUI, "eval.jsonl")
K1, B = 1.5, 0.75

PALABRA = re.compile(r"[0-9a-záéíóúüñ]+(?:[.,][0-9]+)*")


def termina(texto):
    return PALABRA.findall(texto.lower())


def trocear(texto, tam=800, solape=150):
    """El troceado del capítulo 6, literal."""
    trozos, i = [], 0
    while True:
        trozos.append(texto[i:i + tam])
        if i + tam >= len(texto):
            break
        i += tam - solape
    return trozos


def fragmentos(por_documento_completo=False):
    from pypdf import PdfReader

    fragmentos, metadatos = [], []
    for ruta in sorted(glob.glob(PDFS)):
        paginas = [p.extract_text() for p in PdfReader(ruta).pages]
        if por_documento_completo:
            partes = [("\n".join(paginas), 0)]
        else:
            partes = [(t, n) for n, t in enumerate(paginas, start=1)]
        for texto, n_pag in partes:
            for trozo in trocear(texto):
                fragmentos.append(trozo)
                metadatos.append({"documento": os.path.basename(ruta), "pagina": n_pag})
    return fragmentos, metadatos


class BM25:
    def __init__(self, documentos):
        self.docs = [termina(d) for d in documentos]
        self.n = len(self.docs)
        self.largos = [len(d) for d in self.docs]
        self.medio = sum(self.largos) / self.n
        self.frec = [Counter(d) for d in self.docs]
        aparece = Counter()
        for d in self.docs:
            aparece.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in aparece.items()}

    def puntua(self, consulta):
        terminos = termina(consulta)
        notas = []
        for i in range(self.n):
            s = 0.0
            for t in terminos:
                f = self.frec[i].get(t, 0)
                if not f:
                    continue
                s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.largos[i] / self.medio))
            notas.append(s)
        return notas

    def busca(self, consulta, k):
        notas = self.puntua(consulta)
        return sorted(range(self.n), key=lambda i: -notas[i])[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--documento-completo", action="store_true",
                    help="trocear el texto corrido de cada PDF en vez de cada página")
    args = ap.parse_args()

    frags, _ = fragmentos(args.documento_completo)
    print(f"{len(frags)} fragmentos "
          f"({'documento completo' if args.documento_completo else 'página a página'})")

    casos = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    casos = [c for c in casos if c["tipo"] == "factual"]
    print(f"{len(casos)} preguntas factuales con evidencia anotada")

    bm = BM25(frags)
    for k in (1, 4, 10):
        aciertos = sum(any(c["evidencia"] in frags[i] for i in bm.busca(c["pregunta"], k))
                       for c in casos)
        print(f"Recall@{k}: {aciertos / len(casos):.1%} ({aciertos}/{len(casos)})")


if __name__ == "__main__":
    main()
