#!/usr/bin/env python3
"""Mide el recuperador del libro contra la línea base BM25 del apartado 8.3.

Reutiliza `fragmentos()`, `eval.jsonl` y el criterio de acierto del propio
`linea_base_bm25.py`: si se cambia cualquiera de los tres, la comparación deja de
significar nada. Lo único que cambia es CÓMO se busca.

Cuatro recuperadores sobre los mismos 43 fragmentos y las mismas 60 preguntas:

  bm25      la línea base, tal cual (control: tiene que dar 30,0 / 50,0 / 63,3)
  denso     embeddings `intfloat/multilingual-e5-base` con sus prefijos
            `query:` y `passage:`, que es lo que hace `recuperador.py`
  hibrido   fusión recíproca de rangos (RRF, k=60) entre bm25 y denso
  rerank    denso recupera 20 y un cross-encoder multilingüe reordena

Uso:  python mide_recuperador.py [--documento-completo]
"""
import argparse, json, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
import linea_base_bm25 as lb                      # noqa: E402

MODELO_EMB = "intfloat/multilingual-e5-base"      # el de indexar.py
MODELO_RER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
KS = (1, 4, 10)


def recall(orden_por_pregunta, casos, frags):
    """Mismo criterio que la línea base: la evidencia anotada, literal, dentro
    de alguno de los k fragmentos devueltos."""
    out = {}
    for k in KS:
        ac = sum(any(c["evidencia"] in frags[i] for i in orden[:k])
                 for c, orden in zip(casos, orden_por_pregunta))
        out[k] = (ac, len(casos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--documento-completo", action="store_true")
    a = ap.parse_args()

    frags, _ = lb.fragmentos(a.documento_completo)
    casos = [json.loads(l) for l in open(lb.EVAL, encoding="utf-8")]
    casos = [c for c in casos if c["tipo"] == "factual"]
    print(f"{len(frags)} fragmentos · {len(casos)} preguntas factuales "
          f"({'documento completo' if a.documento_completo else 'página a página'})\n")

    maxk = max(KS)
    resultados = {}

    bm = lb.BM25(frags)
    orden_bm = [sorted(range(len(frags)), key=lambda i, n=bm.puntua(c["pregunta"]): -n[i])[:maxk]
                for c in casos]
    resultados["bm25 (línea base)"] = recall(orden_bm, casos, frags)

    import numpy as np
    from sentence_transformers import SentenceTransformer, CrossEncoder
    emb = SentenceTransformer(MODELO_EMB)
    V = emb.encode(["passage: " + f for f in frags], normalize_embeddings=True,
                   batch_size=16, show_progress_bar=False)
    Q = emb.encode(["query: " + c["pregunta"] for c in casos], normalize_embeddings=True,
                   batch_size=16, show_progress_bar=False)
    sim = Q @ V.T
    orden_de = [list(np.argsort(-fila)[:maxk]) for fila in sim]
    resultados["denso (e5-base)"] = recall(orden_de, casos, frags)

    # RRF: suma de 1/(k+rango). No necesita que las dos puntuaciones sean
    # comparables, que es justo el problema de mezclar BM25 con un coseno.
    def rrf(a, b, kk=60):
        p = {}
        for lista in (a, b):
            for r, i in enumerate(lista):
                p[i] = p.get(i, 0.0) + 1.0 / (kk + r + 1)
        return sorted(p, key=lambda i: -p[i])[:maxk]
    orden_bm_l = [sorted(range(len(frags)), key=lambda i, n=bm.puntua(c["pregunta"]): -n[i])[:60]
                  for c in casos]
    orden_de_l = [list(np.argsort(-fila)[:60]) for fila in sim]
    resultados["híbrido (RRF)"] = recall([rrf(x, y) for x, y in zip(orden_bm_l, orden_de_l)],
                                         casos, frags)

    ce = CrossEncoder(MODELO_RER)
    orden_rr = []
    for c, cand in zip(casos, [list(np.argsort(-f)[:20]) for f in sim]):
        notas = ce.predict([(c["pregunta"], frags[i]) for i in cand], show_progress_bar=False)
        orden_rr.append([cand[j] for j in np.argsort(-np.asarray(notas))][:maxk])
    resultados["denso + reranking"] = recall(orden_rr, casos, frags)

    anchura = max(len(n) for n in resultados)
    print(f"{'recuperador':<{anchura}} " + "  ".join(f"Recall@{k:<2}" for k in KS))
    for nombre, r in resultados.items():
        print(f"{nombre:<{anchura}} " +
              "  ".join(f"{r[k][0]/r[k][1]:6.1%} ({r[k][0]:2d}/{r[k][1]})" for k in KS))
    print("\nUn acierto de 60 son 1,7 puntos: diferencias menores son empate.")


if __name__ == "__main__":
    main()
