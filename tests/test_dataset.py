"""Las cifras que el libro publica del dataset, convertidas en pruebas.

La regla es que ninguna cifra impresa en el libro se quede sin una prueba que la
vigile. Si alguien añade un párrafo a un documento de la cooperativa y el troceado
pasa de 43 a 44 fragmentos, el apartado 6.2 del libro queda desmentido por su propio
repositorio: eso tiene que fallar aquí y no descubrirse leyendo.
"""
import json
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS = os.path.join(RAIZ, "datos", "cooperativa")
sys.path.insert(0, RAIZ)


def test_el_validador_del_dataset_pasa():
    """La segunda capa del anexo B, entera, como una sola prueba."""
    r = subprocess.run([sys.executable, os.path.join(RAIZ, "herramientas", "valida_dataset.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_los_pdf_se_regeneran_identicos():
    """Los PDF son un derivado del texto fuente, y el derivado no se edita a mano."""
    antes = {}
    for nombre in sorted(os.listdir(os.path.join(DATOS, "documentos"))):
        if nombre.endswith(".pdf"):
            with open(os.path.join(DATOS, "documentos", nombre), "rb") as f:
                antes[nombre] = f.read()
    r = subprocess.run([sys.executable, os.path.join(DATOS, "genera_pdf.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for nombre, contenido in antes.items():
        with open(os.path.join(DATOS, "documentos", nombre), "rb") as f:
            assert f.read() == contenido, f"{nombre} no se regenera igual"


def test_el_sft_se_regenera_identico():
    with open(os.path.join(DATOS, "sft.jsonl"), encoding="utf-8") as f:
        antes = f.read()
    r = subprocess.run([sys.executable, os.path.join(DATOS, "genera_sft.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(os.path.join(DATOS, "sft.jsonl"), encoding="utf-8") as f:
        assert f.read() == antes


def test_particion_del_capitulo_7():
    """El capítulo 7 imprime «450 ejemplos de entrenamiento y 50 de validación»."""
    with open(os.path.join(DATOS, "sft.jsonl"), encoding="utf-8") as f:
        n = sum(1 for l in f if l.strip())
    validacion = round(n * 0.1)          # test_size=0.1
    assert (n - validacion, validacion) == (450, 50)


def test_linea_base_bm25():
    """Las cifras del apartado 8.3, medidas por el script del propio repositorio.

    Si esta prueba falla después de tocar los documentos o las preguntas, no se
    «arregla» cambiando el número de abajo: hay que decidir si lo que cambia es el
    dataset o lo que dice el libro, y dejar las dos cosas de acuerdo.
    """
    guion = os.path.join(DATOS, "linea_base_bm25.py")
    r = subprocess.run([sys.executable, guion], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "43 fragmentos" in r.stdout
    assert "Recall@1: 30.0% (18/60)" in r.stdout
    assert "Recall@4: 50.0% (30/60)" in r.stdout
    assert "Recall@10: 63.3% (38/60)" in r.stdout

    r = subprocess.run([sys.executable, guion, "--documento-completo"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "41 fragmentos" in r.stdout
    assert "Recall@4: 51.7% (31/60)" in r.stdout


def test_las_cinco_de_inyeccion_traen_su_comprobacion():
    """El apartado 19.3 las añade a la campaña de ataques: sin comprobación no sirven."""
    with open(os.path.join(DATOS, "eval.jsonl"), encoding="utf-8") as f:
        ev = [json.loads(l) for l in f if l.strip()]
    inyeccion = [c for c in ev if c["tipo"] == "inyeccion"]
    assert len(inyeccion) == 5
    for c in inyeccion:
        assert c.get("comprobacion"), c["id"]
        assert c.get("ataque"), c["id"]
    assert sum("documento_inyectado" in c for c in inyeccion) >= 1, (
        "sin documento inyectado no se puede probar la inyección indirecta")
