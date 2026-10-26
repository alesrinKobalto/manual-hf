"""Piezas comunes de los cuadernos de «Crea tu propia IA»."""
from pathlib import Path

LICENCIA = Path(__file__).with_name("_licencia.md").read_text(encoding="utf-8")

def md(texto):  return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}
def code(texto):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.rstrip().splitlines(keepends=True)}

def portada(titulo, capitulos, resumen, requisitos):
    return [md(f"""# {titulo}

{capitulos} del libro **Crea tu propia IA con modelos de código abierto de Hugging Face**,
de Alejandra Espinosa (Kobalto Educación). El libro está en
[kobalto.es/biblioteca](https://kobalto.es/biblioteca/).

{resumen}

**Qué necesitas.** {requisitos}

Los identificadores de modelo se comprueban contra el Hub; donde el libro cita uno
que ha sido retirado, aquí encontrarás el sustituto y una nota explicando el cambio.
"""), md(LICENCIA)]

def cuaderno(celdas):
    return {"cells": celdas, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
