#!/usr/bin/env python3
"""Cuaderno de los capítulos 9, 15 y 17: desplegar, agentes con MCP y ONNX."""
import json, sys
sys.path.insert(0, ".")
from _cabecera import portada, md, code, cuaderno

celdas = portada(
    "Desplegar, agentes con MCP y ONNX",
    "Capítulos 9, 15 y 17",
    "Tres formas de que el asistente salga de tu portátil: una interfaz que puede abrir "
    "cualquiera, un **agente** que llama a herramientas de verdad a través de MCP, y el "
    "modelo convertido a **ONNX** para que corra en el navegador o en un móvil.",
    "La mayor parte corre en CPU. Solo la celda del clasificador necesita el modelo del "
    "capítulo 12, y las de despliegue se ejecutan en tu máquina o en un Space, no aquí.")

celdas += [
md("""## Cómo leer este cuaderno

No todo esto se puede ejecutar dentro de un cuaderno, y conviene decirlo antes de que te
pelees con una celda que no va a funcionar nunca aquí:

| | |
|---|---|
| **Se ejecuta aquí** | la exportación a ONNX, la cuantización y la inferencia con `onnxruntime` |
| **Se ejecuta aquí, pero abre un servidor** | la interfaz de Gradio: arranca y te da una URL temporal |
| **No se ejecuta aquí** | `vllm serve` y el servidor MCP, que son procesos de línea de comandos; el código está para que lo copies a un fichero |

Las celdas del tercer grupo van marcadas y con el código dentro de una cadena, para que puedas
copiarlo sin que el cuaderno intente ejecutarlo."""),

md("""## 1. Instalación"""),
code("""%pip install -q "gradio==6.30.0" "onnx" "onnxruntime==1.31.0" \\
    "transformers==5.19.0" "optimum" "accelerate"
"""),

md("""# Capítulo 9 · Desplegar tu IA

## 2. Una interfaz que puede abrir cualquiera

Gradio monta una interfaz de chat sobre una función de Python. Es lo más corto que hay entre
«tengo un modelo que responde» y «puedo enseñárselo a alguien».

`share=True` da una URL pública temporal: sirve para validar con un cliente, **no para
producción** — caduca y pasa por los servidores de Gradio."""),
code('''import gradio as gr

def responder(mensaje, historial):
    # En el libro, esta función es la del capítulo 6 (RAG). Aquí, un eco
    # para que la celda arranque sin modelo cargado.
    return f"He entendido: «{mensaje}». (Aquí iría la respuesta del asistente.)"

demo = gr.ChatInterface(
    responder,
    title="Asistente de la cooperativa",
    description="Resuelve dudas sobre facturas y tarifas a partir de los documentos internos.",
)
# demo.launch(share=True)     # descomenta para abrirlo; en Kaggle abre una URL temporal
print("interfaz construida:", type(demo).__name__)
'''),
md("""**Qué deberías ver.** `interfaz construida: ChatInterface`. Si descomentas `launch`,
Gradio imprime una URL local y otra pública que dura unas horas."""),

md("""## 3. Cuando la demo deja de servir

Para varios usuarios a la vez hace falta un servidor de inferencia de verdad. **vLLM** expone
una API compatible con la de OpenAI, así que cualquier cliente que ya hables sirve.

Esto es un proceso de línea de comandos: no se lanza desde un cuaderno."""),
code('''DESPLIEGUE = """
# En una máquina con GPU, no en este cuaderno:
vllm serve tu-usuario/asistente-cooperativa --max-model-len 8192 --port 8000

# Y desde cualquier cliente de la API de OpenAI:
#   base_url="http://localhost:8000/v1"
#   model="tu-usuario/asistente-cooperativa"
"""
print(DESPLIEGUE)
'''),
md("""**Ojo con el nombre.** El repositorio se llama `asistente-cooperativa`, **sin sufijo de
tamaño**. Ponerle `-7b` lo deja envejecido en cuanto cambias de modelo base — y es lo que pasó
en la primera edición de este cuaderno."""),

md("""# Capítulo 15 · Agentes y MCP

## 4. Qué es MCP y por qué no es «otro formato de herramientas»

Un agente que llama a funciones las tiene **dentro** de su propio proceso. MCP las saca fuera:
el servidor expone herramientas por un protocolo, y el agente las descubre y las llama sin
saber cómo están implementadas ni en qué lenguaje.

Lo que eso cambia, y es lo que importa para la seguridad: **la única línea que produce efectos
es la del servidor a su sistema, y no pasa por el modelo**. Aunque un documento manipulado
convenciera al agente de pedir algo indebido, el servidor sigue validando.

El servidor es un fichero aparte. Cópialo a `servidor_cooperativa.py`."""),
code('''SERVIDOR = r"""
# servidor_cooperativa.py
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

servidor = MCPServer("cooperativa")
INCIDENCIAS = {}

@servidor.tool()
def consultar_factura(contrato: str, mes: str) -> dict:
    \\"\\"\\"Devuelve el importe de una factura. El agente lee esta descripción.\\"\\"\\"
    if not contrato.isdigit():
        raise ToolError("El número de contrato debe ser numérico")
    return {"contrato": contrato, "mes": mes, "importe_eur": 48.20}

@servidor.tool()
def proponer_incidencia(contrato: str, motivo: str) -> dict:
    \\"\\"\\"PROPONE una incidencia: no la crea. La valida una persona.\\"\\"\\"
    return {"estado": "propuesta", "contrato": contrato, "motivo": motivo}

if __name__ == "__main__":
    servidor.run()
"""
open("servidor_cooperativa.py", "w").write(SERVIDOR.strip())
print("escrito servidor_cooperativa.py ·", len(SERVIDOR.splitlines()), "líneas")
'''),
md("""**Atención a la versión.** En `mcp` 2.x la clase `FastMCP` pasó a llamarse `MCPServer`.
El código de tutoriales anteriores (`from mcp.server.fastmcp import FastMCP`) falla con un
`ImportError` seco. Es el tipo de cambio del que avisa el prólogo del libro.

Y fíjate en `proponer_incidencia`: **propone, no crea**. Esa es la frontera. Una herramienta que
escribe en el sistema de verdad no se le da a un agente sin que una persona valide."""),

md("""## 5. El cliente

Levanta el servidor como subproceso, descubre las herramientas y se las entrega al agente."""),
code('''CLIENTE = r"""
# cliente.py
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["servidor_cooperativa.py"])
    async with stdio_client(params) as (lectura, escritura):
        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()
            herramientas = await sesion.list_tools()
            for h in herramientas.tools:
                print("-", h.name, "·", h.description.splitlines()[0])
            print(await sesion.call_tool("consultar_factura",
                                         {"contrato": "10482", "mes": "agosto"}))

asyncio.run(main())
"""
open("cliente.py", "w").write(CLIENTE.strip())
print("escrito cliente.py")
'''),
md("""**Qué deberías ver al ejecutarlo en tu máquina** (`python cliente.py`): las dos
herramientas con su descripción, y el resultado de la llamada. Aquí no se ejecuta porque
`mcp` no viene en la imagen de Kaggle y levantar subprocesos desde un cuaderno da más
problemas que valor."""),

md("""## 6. Enganchar el agente

`smolagents` lee las herramientas del servidor y las pone a disposición del modelo. La versión
importa: `mcpadapt`, que es quien hace de puente, todavía no admite `mcp` 2.x."""),
code('''print("""pip install "smolagents[mcp]==1.26.0" "mcp<2"

from smolagents import ToolCollection, CodeAgent, InferenceClientModel
from mcp import StdioServerParameters
import sys

params = StdioServerParameters(command=sys.executable, args=["servidor_cooperativa.py"])
with ToolCollection.from_mcp(params, trust_remote_code=False) as coleccion:
    agente = CodeAgent(tools=[*coleccion.tools], model=InferenceClientModel())
    print(agente.run("¿Cuánto pagó en agosto el contrato 10482?"))
""")
'''),
md("""**El `mcp<2` de ahí arriba no es un descuido.** `mcpadapt` no admite todavía la 2.x, y la
integración de agentes pasa por él. Revísalo en cada edición: es exactamente el tipo de
incompatibilidad entre dos librerías que se arregla sola un día y rompe el código otro."""),

md("""# Capítulo 17 · IA en el navegador y en el dispositivo

## 7. Por qué ONNX

Un modelo en PyTorch necesita Python y la librería instalada. **ONNX** es un formato intermedio
que ejecutan runtimes de C++, de JavaScript o de móvil: el mismo modelo corre en el navegador
del usuario, sin servidor y sin que sus datos salgan de su máquina.

**Esta sección reproduce la tabla del capítulo 17**, y para eso usa exactamente el modelo que
el libro declara medir: *clasificador tipo BERT de 4 capas y dimensión 256, pesos aleatorios,
32 tokens*. No es un modelo del Hub: es sintético, y eso es deliberado — una medición de
formato y de latencia no necesita pesos entrenados, y así cualquiera la repite sin descargar
nada.

**Y conviene medirlo en tu modelo, no darlo por hecho.** Dos modelos con un reparto interno
opuesto dan la misma cuarta parte: este, que es **73 % *embeddings***, y BETO
(`bert-base-spanish-wwm-cased`), que es solo **22 %** — medido al montar este cuaderno:
439,6 MB en fp32 y 110,6 en int8, el **25,2 %**. Si la cuantización se saltara los
*embeddings*, BETO se habría quedado en el 41 % y este en el 80 %. Que los dos den la cuarta
parte dice que esta ruta de exportación los cuantiza también."""),
code('''import os, time, torch, numpy as np
from transformers import BertConfig, BertForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

torch.manual_seed(0)
config = BertConfig(          # exactamente lo que declara el libro
    num_hidden_layers=4, hidden_size=256, num_attention_heads=4,
    intermediate_size=1024, vocab_size=32000, num_labels=4,   # ver la nota de abajo
)
modelo = BertForSequenceClassification(config).eval()
emb = sum(p.numel() for n, p in modelo.named_parameters() if "embeddings" in n)
tot = sum(p.numel() for p in modelo.parameters())
print(f"{tot/1e6:.2f} M parámetros · embeddings {emb/tot*100:.1f} %")

ejemplo = {"input_ids": torch.randint(0, config.vocab_size, (1, 32)),
           "attention_mask": torch.ones(1, 32, dtype=torch.long)}

os.makedirs("onnx", exist_ok=True)      # torch.onnx.export NO crea la carpeta
torch.onnx.export(
    modelo, (ejemplo["input_ids"], ejemplo["attention_mask"]), "onnx/model.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "lote", 1: "secuencia"},
                  "attention_mask": {0: "lote", 1: "secuencia"}},
    opset_version=17, dynamo=False,
)
quantize_dynamic("onnx/model.onnx", "onnx/model_quantized.onnx", weight_type=QuantType.QInt8)

def tam(p):                      # MB decimales y MiB: la diferencia es del 5 %
    n = os.path.getsize(p)
    return n / 1e6, n / 2**20
a_mb, a_mib = tam("onnx/model.onnx")
b_mb, b_mib = tam("onnx/model_quantized.onnx")
print(f"ONNX fp32: {a_mb:.1f} MB = {a_mib:.1f} MiB   (el libro mide 44,2)")
print(f"ONNX int8: {b_mb:.1f} MB = {b_mib:.1f} MiB   (el libro mide 11,2)")
print(f"proporción: {b_mb/a_mb*100:.1f} %   (el libro, 25,3 %)")
'''),
md("""**Qué deberías ver.** Unos **11,55 millones de parámetros**, de los cuales un **73 % son
*embeddings***, y los dos tamaños muy cerca de los del libro.

**El vocabulario es el único dato que el libro no declara**, y es el que decide el tamaño: con
32.000 entradas la cuenta sale a 44,06 MiB contra los 44,2 de la tabla, tres décimas por
ciento. Con 33.000 se va un 2 % arriba. Por eso la celda imprime **MB y MiB**: la diferencia
entre las dos unidades es del 5 % y basta para desalinear la comparación. La cuarta parte sale de ahí: 11,6 M ×
4 bytes ≈ 44 MB en fp32 y × 1 byte ≈ 11 en int8.

**Y ese 73 % es la razón por la que la cifra importa.** Si la cuantización se saltara los
*embeddings* —como pasa con algunas rutas de exportación—, el fichero se quedaría en torno al
80 % del original, no en el 25 %. Que el libro mida 11,2 significa que su exportación los
cuantiza también.

**El `os.makedirs` no sobra**: `torch.onnx.export` no crea la carpeta de destino y falla con un
error de fichero no encontrado que no dice que el problema sea la carpeta."""),

md("""## 8. Ejecutar, medir y comprobar que da lo mismo

Lo que importa no es solo que corra: es que **dé el mismo resultado**. La tabla del libro
compara los logits de las tres versiones y la clase ganadora, no solo la latencia."""),
code('''import onnxruntime as ort

entrada = {k: v.numpy() for k, v in ejemplo.items()}
with torch.no_grad():
    ref = modelo(**ejemplo).logits.numpy()

def medir(ruta, repeticiones=50):
    s = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    s.run(["logits"], entrada)                       # calentar
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        out = s.run(["logits"], entrada)[0]
    return out, (time.perf_counter() - t0) / repeticiones * 1000

for nombre, ruta in (("fp32", "onnx/model.onnx"), ("int8", "onnx/model_quantized.onnx")):
    try:
        out, ms = medir(ruta)
        print(f"ONNX {nombre}: {ms:5.2f} ms · dif. máx. con PyTorch {np.abs(out-ref).max():.4f}"
              f" · misma clase: {'sí' if out.argmax()==ref.argmax() else 'NO'}")
    except Exception as e:
        print(f"ONNX {nombre}: {type(e).__name__} — {str(e)[:120]}")

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(50): modelo(**ejemplo)
print(f"PyTorch  : {(time.perf_counter()-t0)/50*1000:5.2f} ms (referencia)")
'''),
md("""**Qué deberías ver.** Las dos versiones de ONNX más rápidas que PyTorch, la diferencia de
logits del fp32 en 0,0000 y la del int8 en milésimas, y la misma clase en las dos. **Las
latencias no van a coincidir con las del libro**: él midió en una CPU de cuatro núcleos y aquí
depende de la máquina que toque. Lo que sí tiene que coincidir es el orden —ONNX por debajo de
PyTorch— y que la clase no cambie.

**Y la celda está dentro de un `try` a propósito.** Un fichero cuantizado puede escribirse sin
error y luego no cargarse: probando otra arquitectura al montar este cuaderno, `onnxruntime`
devolvió `InvalidGraph` sobre un fichero que se había escrito sin dar ningún problema.
**Escribirse y cargarse son dos cosas distintas**, y la única forma de saberlo es
intentarlo."""),

md("""## 9. Y lo mismo en el navegador

`transformers.js` espera **la misma estructura que un repositorio del Hub**: `config.json`, los
ficheros del tokenizador y una carpeta `onnx/` con `model.onnx` y `model_quantized.onnx`. Si
copias solo el `.onnx` suelto, no lo carga.

Esto es JavaScript: se ejecuta con Node o en una página, no en este cuaderno."""),
code('''print("""// clasificar.mjs  (Node 22, @huggingface/transformers 4.3.1)
import { pipeline, env } from "@huggingface/transformers";

env.allowRemoteModels = false;          // cargar solo modelos locales
env.localModelPath = "./modelos/";

for (const dtype of ["fp32", "q8"]) {
  const clasificador = await pipeline("text-classification", "clasificador", { dtype });
  console.log(dtype, JSON.stringify(await clasificador("se ha roto la tuberia del bano")));
}
""")
'''),
md("""**Por qué el ejemplo va sin tildes.** No es un descuido: comprobar que el modelo se
comporta igual con texto sin acentuar es parte de probarlo, porque mucho texto real de
formularios llega así."""),
]

json.dump(cuaderno(celdas), open("agentes-mcp-y-despliegue.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("escrito agentes-mcp-y-despliegue.ipynb")
