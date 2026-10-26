"""Lo que el libro promete del repositorio, no del dataset."""
import os
import re
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Las once versiones que el libro fija en sus primeras páginas.
VERSIONES_DEL_LIBRO = {
    "transformers": "5.19", "trl": "1.15", "peft": "0.21", "datasets": "5.1",
    "sentence-transformers": "6.1", "smolagents": "1.26", "diffusers": "0.41",
    "gradio": "6.30", "mcp": "2.3", "onnxruntime": "1.31",
}
SCRIPTS = ["indexar.py", "entrenar.py", "evaluar.py", "app.py", "recuperador.py"]


def test_requirements_fija_las_versiones_del_libro():
    with open(os.path.join(RAIZ, "requirements.txt"), encoding="utf-8") as f:
        texto = f.read()
    fijadas = dict(re.findall(r"^([a-z0-9-]+)==([0-9.]+)", texto, re.M))
    for paquete, serie in VERSIONES_DEL_LIBRO.items():
        assert paquete in fijadas, f"{paquete} no está fijado en requirements.txt"
        assert fijadas[paquete].startswith(serie + "."), (
            f"{paquete} fijado en {fijadas[paquete]}, y el libro dice {serie}")


def test_transformers_js_no_se_instala_con_pip():
    """Es la undécima versión del libro y es JavaScript: si aparece aquí, es un error."""
    with open(os.path.join(RAIZ, "requirements.txt"), encoding="utf-8") as f:
        lineas = [l for l in f if not l.strip().startswith("#")]
    assert not any("transformers.js" in l or "transformers-js" in l for l in lineas)


def test_los_dos_entornos_de_mcp_no_se_contradicen():
    """requirements.txt fija mcp 2.3; el de agentes exige mcp<2. Separados a propósito."""
    with open(os.path.join(RAIZ, "requirements.txt"), encoding="utf-8") as f:
        principal = f.read()
    with open(os.path.join(RAIZ, "requirements-agentes.txt"), encoding="utf-8") as f:
        agentes = f.read()
    assert "mcp==2.3.0" in principal
    assert "mcp<2" in agentes
    assert "smolagents[mcp]==1.26.0" in agentes


def test_los_scripts_compilan():
    for nombre in SCRIPTS:
        r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(RAIZ, nombre)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{nombre}: {r.stderr}"


def test_los_cuadernos_coinciden_con_su_fuente():
    r = subprocess.run([sys.executable, os.path.join(RAIZ, "herramientas", "generar_cuadernos.py"),
                        "--verifica"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_las_licencias_estan_completas():
    """El anexo B dice que los textos legales completos están en el repositorio."""
    esperado = {"apache-2.0.txt": "Apache License",
                "cc-by-4.0.txt": "Attribution 4.0 International",
                "cc-by-sa-4.0.txt": "Attribution-ShareAlike 4.0 International"}
    for nombre, primera in esperado.items():
        ruta = os.path.join(RAIZ, "licencias", nombre)
        assert os.path.exists(ruta), f"falta {nombre}"
        with open(ruta, encoding="utf-8") as f:
            contenido = f.read()
        assert contenido.lstrip().startswith(primera), nombre
        assert len(contenido) > 10000, f"{nombre} parece truncado"


def test_la_plantilla_de_errata_pide_los_cuatro_campos():
    """El anexo B: «indicando capítulo, texto actual, problema y propuesta»."""
    with open(os.path.join(RAIZ, ".github", "ISSUE_TEMPLATE", "errata.yml"), encoding="utf-8") as f:
        plantilla = f.read()
    for campo in ("id: capitulo", "id: actual", "id: problema", "id: propuesta"):
        assert campo in plantilla, campo
