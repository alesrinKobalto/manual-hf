#!/usr/bin/env python3
"""Regenera los cuadernos de `cuadernos/` desde su fuente, y comprueba que no han cambiado.

El anexo B lo pide para toda propuesta que cambie el libro: «regenerar los cuadernos
con herramientas/generar_cuadernos.py».

    python herramientas/generar_cuadernos.py            # regenera y avisa de los cambios
    python herramientas/generar_cuadernos.py --verifica  # no escribe: falla si algo cambió

Cada cuaderno se construye desde un `.py` que solo monta celdas con `_cabecera.py`: no
ejecuta modelos ni descarga nada, así que corre en dos segundos y vale para la
integración continua. La generación es determinista, y por eso `--verifica` puede
comparar byte a byte: si el fuente y el `.ipynb` del repositorio discrepan, es que
alguien editó el cuaderno a mano y ese cambio se perdería en la siguiente generación.

`kobalto-hf-ejemplos.ipynb` **no tiene fuente** y se informa de ello en cada ejecución:
está publicado y versionado aquí, pero se escribió antes de que existieran los
generadores y todavía no se ha convertido. Mientras eso siga así, ese cuaderno hay que
editarlo a mano.
"""
import argparse
import hashlib
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUADERNOS = os.path.join(RAIZ, "cuadernos")

# fuente .py -> cuaderno que escribe
FUENTES = {
    "fundamentos.py": "kobalto-hf-fundamentos.ipynb",
    "rag.py": "rag-conectar-la-ia-con-tus-documentos.ipynb",
    "inferencia.py": "inferencia-local-y-cuantizacion.ipynb",
    "cap7_ajuste_fino.py": "cap7-ajuste-fino-con-lora.ipynb",
    "agentes_mcp.py": "agentes-mcp-y-despliegue.ipynb",
}
SIN_FUENTE = ["kobalto-hf-ejemplos.ipynb"]


def huella(ruta):
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verifica", action="store_true",
                    help="no escribir: fallar si algún cuaderno no coincide con su fuente")
    args = ap.parse_args()

    antes = {n: huella(os.path.join(CUADERNOS, n))
             for n in FUENTES.values() if os.path.exists(os.path.join(CUADERNOS, n))}

    cambiados, fallos = [], []
    for fuente, cuaderno in FUENTES.items():
        r = subprocess.run([sys.executable, fuente], cwd=CUADERNOS,
                           capture_output=True, text=True)
        if r.returncode:
            fallos.append(f"{fuente}: {r.stderr.strip().splitlines()[-1] if r.stderr else 'error'}")
            continue
        ruta = os.path.join(CUADERNOS, cuaderno)
        nueva = huella(ruta)
        if cuaderno in antes and antes[cuaderno] != nueva:
            cambiados.append(cuaderno)
            print(f"  CAMBIA  {cuaderno}")
        else:
            print(f"  igual   {cuaderno}")

    for cuaderno in SIN_FUENTE:
        print(f"  sin fuente  {cuaderno} (se edita a mano)")

    print()
    if fallos:
        for f_ in fallos:
            print("FALLA", f_)
        return 1
    if args.verifica and cambiados:
        print(f"{len(cambiados)} cuaderno(s) no coinciden con su fuente: "
              f"{', '.join(cambiados)}.\nEjecute `python herramientas/generar_cuadernos.py` "
              f"y revise el diff antes de subir.")
        return 1
    if cambiados:
        print(f"{len(cambiados)} cuaderno(s) regenerados. Revise el diff antes de subir.")
    else:
        print("Los cuadernos coinciden con sus fuentes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
