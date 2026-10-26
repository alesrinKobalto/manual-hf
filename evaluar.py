#!/usr/bin/env python3
"""Capítulo 8: medir el recuperador y, si se pide, puntuar las respuestas.

    python evaluar.py                 # Recall@1, @4 y @10 sobre las 60 factuales
    python evaluar.py --responder      # además genera respuestas con el modelo
    python evaluar.py --k 4 --tipo sin_respuesta

Primero se mide el recuperador **por separado**: en un RAG, la mitad de los errores
vienen de recuperar el fragmento equivocado, no del LLM, y si el Recall@4 es bajo
ningún cambio de modelo lo arregla.

La línea base contra la que comparar no necesita ni modelos ni GPU:
`python datos/cooperativa/linea_base_bm25.py`.

El conjunto de evaluación no se usa nunca para entrenar. `herramientas/valida_dataset.py`
lo comprueba con el umbral de contaminación del capítulo 16.
"""
import argparse
import json
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(RAIZ, "datos", "cooperativa", "eval.jsonl")


def casos(tipo=None):
    with open(EVAL, encoding="utf-8") as f:
        todos = [json.loads(l) for l in f if l.strip()]
    return [c for c in todos if tipo is None or c["tipo"] == tipo]


def recall(recuperador, factuales, ks=(1, 4, 10)):
    """Recall@k del apartado 8.3: ¿está la evidencia en alguno de los k recuperados?"""
    resultados = {}
    for k in ks:
        aciertos = 0
        for c in factuales:
            recuperados = recuperador.busca(c["pregunta"], k)
            aciertos += any(c["evidencia"] in t for t, _, _ in recuperados)
        resultados[k] = aciertos
        print(f"Recall@{k}: {aciertos / len(factuales):.1%} ({aciertos}/{len(factuales)})")
    return resultados


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, nargs="*", default=[1, 4, 10])
    ap.add_argument("--tipo", default="factual",
                    help="factual, sin_respuesta, ambigua, fuera_de_ambito o inyeccion")
    ap.add_argument("--responder", action="store_true",
                    help="generar la respuesta con el modelo además de medir el recuperador")
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adaptador", default=None, help="carpeta del adaptador LoRA entrenado")
    args = ap.parse_args()

    from recuperador import Recuperador, contexto_y_mensajes

    rec = Recuperador()
    seleccion = casos(args.tipo)
    print(f"{len(seleccion)} preguntas de tipo {args.tipo}\n")

    if args.tipo == "factual":
        recall(rec, seleccion, tuple(args.k))

    if not args.responder:
        return

    from transformers import pipeline
    chat = pipeline("text-generation", model=args.modelo, device_map="auto")
    if args.adaptador:
        chat.model.load_adapter(args.adaptador)

    print()
    for c in seleccion:
        recuperados = rec.busca(c["pregunta"], max(args.k))
        mensajes = contexto_y_mensajes(c["pregunta"], recuperados)
        salida = chat(mensajes, max_new_tokens=300, do_sample=False)[0]["generated_text"][-1]
        print(f"— {c['id']} {c['pregunta']}")
        print(f"  {salida['content'].strip()}")
        if "respuesta" in c:
            print(f"  (referencia: {c['respuesta']})")
        print()


if __name__ == "__main__":
    main()
