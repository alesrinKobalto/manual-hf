#!/usr/bin/env python3
"""Capítulo 6: trocear los PDF de la cooperativa y calcular sus embeddings.

Primera fase de las dos que tiene un RAG. Se ejecuta una vez, o cada vez que cambian
los documentos, y deja en disco el índice que usan `evaluar.py` y `app.py`.

    python indexar.py                      # índice página a página (43 fragmentos)
    python indexar.py --documento-completo  # el otro troceado del libro (41)

El modelo de embeddings es `intfloat/multilingual-e5-base`, que **exige** los prefijos
`query:` y `passage:`: sin ellos funciona peor y no avisa. Otros modelos no los usan y
su model card lo indica; es el tipo de detalle que no se adivina.

La primera ejecución descarga unos 1,1 GB de pesos. Después, nada: todo en local.
"""
import argparse
import glob
import json
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))
PDFS = os.path.join(RAIZ, "datos", "cooperativa", "documentos", "*.pdf")
MODELO = "intfloat/multilingual-e5-base"


def trocear(texto, tam=800, solape=150):
    """Troceado del apartado 6.2, con solape para no partir una idea en dos."""
    trozos, i = [], 0
    while True:
        trozos.append(texto[i:i + tam])
        if i + tam >= len(texto):   # el último trozo ya llega al final
            break
        i += tam - solape
    return trozos


def lee_fragmentos(documento_completo=False):
    from pypdf import PdfReader

    fragmentos, metadatos = [], []
    for ruta in sorted(glob.glob(PDFS)):
        paginas = [p.extract_text() for p in PdfReader(ruta).pages]
        partes = ([("\n".join(paginas), 0)] if documento_completo
                  else [(t, n) for n, t in enumerate(paginas, start=1)])
        for texto, n_pag in partes:
            for trozo in trocear(texto):
                fragmentos.append(trozo)
                metadatos.append({"documento": os.path.basename(ruta), "pagina": n_pag})
    return fragmentos, metadatos


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--documento-completo", action="store_true",
                    help="trocear el texto corrido de cada PDF en vez de cada página")
    ap.add_argument("--salida", default=os.path.join(RAIZ, "indice"),
                    help="carpeta donde se guardan los vectores y los metadatos")
    args = ap.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    fragmentos, metadatos = lee_fragmentos(args.documento_completo)
    print(f"{len(fragmentos)} fragmentos de {len(set(m['documento'] for m in metadatos))} "
          f"documentos; el primero es {metadatos[0]}")

    emb = SentenceTransformer(MODELO)
    vectores = emb.encode(["passage: " + f for f in fragmentos],
                          normalize_embeddings=True, show_progress_bar=True)

    os.makedirs(args.salida, exist_ok=True)
    np.save(os.path.join(args.salida, "vectores.npy"), vectores)
    with open(os.path.join(args.salida, "fragmentos.json"), "w", encoding="utf-8") as f:
        json.dump({"modelo": MODELO, "fragmentos": fragmentos, "metadatos": metadatos},
                  f, ensure_ascii=False)
    print(f"índice guardado en {os.path.relpath(args.salida, RAIZ)}/ "
          f"({vectores.shape[0]} vectores de {vectores.shape[1]} dimensiones)")


if __name__ == "__main__":
    main()
