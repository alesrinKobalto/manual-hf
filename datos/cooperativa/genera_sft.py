#!/usr/bin/env python3
"""Construye `sft.jsonl` a partir de `semillas.json`.

El capítulo 16 describe el proceso real: un experto escribe entre 30 y 50 semillas,
una por intención de usuario, y a partir de ahí se generan variaciones que se filtran
y se deduplican. Aquí las 500 formulaciones y las 150 respuestas están escritas a
mano —ninguna sale de un modelo— y este script solo las empareja, sin azar: la
formulación n de cada intención va con su respuesta n módulo el número de respuestas.
Dos ejecuciones dan el mismo fichero byte a byte.

    python datos/cooperativa/genera_sft.py

`herramientas/valida_dataset.py` comprueba después lo que el libro afirma del
dataset: 500 conversaciones distintas con el umbral 0,6 del capítulo 16 y ninguna
pregunta de evaluación contaminada con el umbral 0,5.
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
SEMILLAS = os.path.join(AQUI, "semillas.json")
DESTINO = os.path.join(AQUI, "sft.jsonl")


def conversaciones():
    with open(SEMILLAS, encoding="utf-8") as f:
        datos = json.load(f)
    sistema = datos["sistema"]
    for intencion in datos["intenciones"]:
        respuestas = intencion["respuestas"]
        for n, pregunta in enumerate(intencion["preguntas"]):
            yield {"messages": [
                {"role": "system", "content": sistema},
                {"role": "user", "content": pregunta},
                {"role": "assistant", "content": respuestas[n % len(respuestas)]},
            ]}


def main():
    with open(DESTINO, "w", encoding="utf-8") as f:
        n = 0
        for conv in conversaciones():
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} conversaciones escritas en {os.path.relpath(DESTINO)}")


if __name__ == "__main__":
    main()
