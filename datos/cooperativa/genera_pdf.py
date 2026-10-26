#!/usr/bin/env python3
"""Genera los seis PDF del dataset a partir de su texto fuente.

Los PDF son binarios: en el repositorio manda el texto de `documentos/fuente/*.txt`,
y este script los reconstruye byte a byte iguales (`invariant=1` fija la fecha interna
del PDF, que si no cambiaría en cada ejecución y ensuciaría el diff).

    python datos/cooperativa/genera_pdf.py

El troceado del capítulo 6 depende de la longitud del texto extraído, y el libro
publica dos cifras que salen de ella: 43 fragmentos troceando página a página y 41
troceando el documento completo. `herramientas/valida_dataset.py` las comprueba.
"""
import os
import sys

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

AQUI = os.path.dirname(os.path.abspath(__file__))
FUENTE = os.path.join(AQUI, "documentos", "fuente")
DESTINO = os.path.join(AQUI, "documentos")

MARCA_PAGINA = "---PAGINA---"
MARGEN_X, ALTO_INICIAL, INTERLINEA = 56, 790, 13.2
FUENTE_TIPO, CUERPO = "Helvetica", 9.8


def paginas(nombre):
    """Las páginas de un documento fuente, cada una como lista de líneas."""
    with open(os.path.join(FUENTE, nombre), encoding="utf-8") as f:
        texto = f.read()
    return [p.strip("\n").split("\n") for p in texto.split(MARCA_PAGINA)]


def escribe_pdf(nombre_txt):
    nombre_pdf = nombre_txt.replace(".txt", ".pdf")
    ruta = os.path.join(DESTINO, nombre_pdf)
    c = canvas.Canvas(ruta, pagesize=A4, invariant=1)
    c.setTitle(nombre_pdf)
    for n, lineas in enumerate(paginas(nombre_txt)):
        if n:
            c.showPage()
        c.setFont(FUENTE_TIPO, CUERPO)
        y = ALTO_INICIAL
        for linea in lineas:
            if linea.strip():
                c.drawString(MARGEN_X, y, linea)
            y -= INTERLINEA
    c.save()
    return nombre_pdf


def main():
    os.makedirs(DESTINO, exist_ok=True)
    fuentes = sorted(f for f in os.listdir(FUENTE) if f.endswith(".txt"))
    if len(fuentes) != 6:
        sys.exit(f"Se esperaban 6 documentos fuente y hay {len(fuentes)}")
    for nombre in fuentes:
        print("escrito", escribe_pdf(nombre))


if __name__ == "__main__":
    main()
