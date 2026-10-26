#!/usr/bin/env python3
"""Compara lo que fija `requirements.txt` con lo que hay hoy en PyPI.

Primera capa del anexo B, la parte que se puede automatizar sin GPU: no comprueba que
el código siga funcionando —eso exige ejecutarlo—, pero avisa de lo único que puede
romperlo sin que nadie toque nada, que es una versión nueva de una librería.

    python herramientas/vigila_versiones.py            # informa y sale con 0
    python herramientas/vigila_versiones.py --estricto  # sale con 1 si hay un salto mayor

Un salto de versión **mayor** es el que ha roto el libro cada vez: los pipelines
retirados en transformers 5, `apply_chat_template` devolviendo otra cosa, el
renombrado de `FastMCP`. Un salto menor se anota; un salto mayor pide revisar el
capítulo antes de la edición siguiente.

Necesita red y nada más: no instala ninguna de las librerías.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQS = [os.path.join(RAIZ, "requirements.txt"),
        os.path.join(RAIZ, "requirements-datos.txt")]


def version(v):
    return [int(x) for x in re.findall(r"\d+", v)[:3]]


def ultima(paquete):
    url = f"https://pypi.org/pypi/{paquete}/json"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["info"]["version"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--estricto", action="store_true",
                    help="salir con código 1 si alguna librería ha subido de versión mayor")
    args = ap.parse_args()

    fijadas = {}
    for ruta in REQS:
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                m = re.match(r"^([a-z0-9._-]+)==([0-9][0-9a-z.]*)", linea.strip())
                if m:
                    fijadas[m.group(1)] = m.group(2)

    mayores, menores, errores = [], [], []
    for paquete, fijada in sorted(fijadas.items()):
        try:
            hoy = ultima(paquete)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            errores.append(f"{paquete}: {e}")
            print(f"  ?       {paquete:22} fijada {fijada:10} (no se pudo consultar)")
            continue
        a, b = version(fijada), version(hoy)
        if b[0] > a[0]:
            mayores.append((paquete, fijada, hoy))
            print(f"  MAYOR   {paquete:22} fijada {fijada:10} en PyPI {hoy}")
        elif b > a:
            menores.append((paquete, fijada, hoy))
            print(f"  menor   {paquete:22} fijada {fijada:10} en PyPI {hoy}")
        else:
            print(f"  al día  {paquete:22} {fijada}")

    print()
    print(f"{len(fijadas)} paquetes fijados · {len(mayores)} salto(s) mayor(es) · "
          f"{len(menores)} menor(es)" + (f" · {len(errores)} sin consultar" if errores else ""))
    if mayores:
        print("\nRevisar antes de la próxima edición, capítulo por capítulo:")
        for paquete, fijada, hoy in mayores:
            print(f"  · {paquete}: {fijada} -> {hoy}")
    return 1 if (args.estricto and mayores) else 0


if __name__ == "__main__":
    sys.exit(main())
