#!/usr/bin/env python3
"""Cuaderno del capítulo 5: inferencia local y cuantización."""
import json, sys
sys.path.insert(0, ".")
from _cabecera import portada, md, code, cuaderno

celdas = portada(
    "Inferencia local y cuantización",
    "Capítulo 5",
    "Cargar un modelo de 7.600 millones de parámetros en una tarjeta donde no cabe, "
    "y medir qué cuesta: memoria, velocidad y si las respuestas siguen valiendo.",
    "**GPU obligatoria.** Vale la T4 gratuita de Kaggle o de Colab. La descarga son "
    "unos 15 GB y tarda unos minutos la primera vez.")

celdas += [
md("""## Qué vas a medir, y qué dice el libro

Este cuaderno **reproduce una medición publicada**. El capítulo 5 declara, para
`Qwen/Qwen2.5-7B-Instruct` en 4 bits sobre una T4 gratuita:

| | el libro |
|---|---|
| memoria residente | **~5,5 GB** |
| velocidad | **10-16 tokens/s** |

Y la aritmética de por qué eso es la noticia: un modelo de **7.600 millones de parámetros** en
16 bits necesita unos **15 GB solo de pesos**, y la T4 tiene **15,6 GB de VRAM**. No cabe,
porque además hace falta sitio para la caché de atención. En 4 bits los pesos bajan a unos
**5,5 GB** —no a los 3,8 teóricos, y la celda 2 explica por qué— y ahí sí.

**Las cifras exactas dependerán de tu máquina.** Lo que tiene que coincidir es el orden de
magnitud de la memoria —unos pocos GB, no quince— y que el modelo responda con sentido, que es
lo que demuestra que cuantizar no lo ha roto."""),

md("""## 1. Instalación"""),
code("""%pip install -q "transformers==5.19.0" "bitsandbytes" "accelerate"
"""),
code('''import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Kaggle ofrece «T4 x2»; aquí basta una
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna")
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("bf16 nativo:", torch.cuda.is_bf16_supported(including_emulation=False))
'''),
md("""**Qué deberías ver.** `Tesla T4`, unos **15,8 GB** de VRAM y `bf16 nativo: False`.

Ese `False` es el que decide la línea siguiente del código: la T4 es anterior a la arquitectura
Ampere y no admite bfloat16 de forma nativa, así que el cálculo va en float16. Elegirlo a mano
es el error que hace que el mismo código funcione en una máquina y falle en otra."""),

md("""## 2. Cargar un 7B que no cabe

`load_in_4bit` no descarga un modelo distinto: descarga los pesos normales y los **cuantiza al
vuelo** al cargarlos. Por eso la descarga sigue siendo de unos 15 GB aunque en memoria ocupe
cuatro veces menos — y por eso esta celda tarda unos minutos la primera vez.

`nf4` es el tipo de cuantización de 4 bits del artículo de QLoRA, pensado para pesos con
distribución normal. `bnb_4bit_compute_dtype` es distinto: es en qué precisión se hacen las
multiplicaciones, no en cuál se guardan los pesos."""),
code('''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

modelo_id = "Qwen/Qwen2.5-7B-Instruct"
# La T4 no admite bfloat16 de forma nativa: se usa float16 en ese caso
tipo = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=tipo,
)
tok = AutoTokenizer.from_pretrained(modelo_id)
modelo = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=bnb, device_map="auto")

cfg = modelo.config
print(f"capas {cfg.num_hidden_layers} · dimensión {cfg.hidden_size} · vocabulario {cfg.vocab_size:,}")
print(f"residente ahora : {torch.cuda.memory_allocated()/1e9:.2f} GB   <- lo que ocupa el modelo")
print(f"pico al cargar  : {torch.cuda.max_memory_allocated()/1e9:.2f} GB   <- incluye los buffers de la cuantización")
'''),
md("""**Qué deberías ver, y conviene saber qué es cada número.**

- **Residente** es lo que ocupa el modelo ya cargado: en torno a **5,5 GB**.
- **Pico al cargar** es mayor, porque cuantizar necesita tener a la vez el peso original y el
  cuantizado. Ese pico es el que decide si cabe, no el residente.

**Y el residente no son los 3,8 GB teóricos de 7,6 B parámetros a 4 bits**, por una razón que
el libro explica y que conviene tener delante: `bitsandbytes` **no cuantiza todo**. Deja en 16
bits la tabla de *embeddings* y la capa de salida, que en este modelo son unos 545 millones de
parámetros cada una. La cuenta:

| | |
|---|---|
| capas lineales cuantizadas | ~6,5 B × 0,5 B = **3,3 GB** |
| *embeddings* + salida, en fp16 | ~1,1 B × 2 B = **2,2 GB** |
| **total esperado** | **~5,5 GB** |

Si tu número se parece a eso, está bien. **Si te sale la mitad, sospecha de la medida**: lo que
se mide tiene que ser `memory_allocated()` después de cargar, no un valor leído antes de que la
carga termine."""),

md("""## 3. Que responda, que es lo que demuestra que no se ha roto

Cuantizar degrada. La pregunta no es si degrada, sino **si degrada lo bastante para que se
note**, y eso solo se ve preguntándole algo que requiera precisión."""),
code('''mensajes = [
    {"role": "system", "content": "Eres un asistente de una comercializadora eléctrica. "
                                  "Respondes en español, claro y en pocas frases."},
    {"role": "user", "content": "¿Qué es el término de potencia y por qué se paga aunque no consuma?"},
]
# En transformers 5, apply_chat_template devuelve un diccionario (input_ids + attention_mask)
entrada = tok.apply_chat_template(mensajes, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(modelo.device)
salida = modelo.generate(**entrada, max_new_tokens=160, do_sample=False)
print(tok.decode(salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True))
'''),
md("""**Qué deberías ver.** Una explicación correcta: que el término de potencia paga la
capacidad reservada, no la energía, y por eso se abona aunque el consumo sea cero.

**Con `do_sample=False` sale siempre lo mismo**, así que si lo ejecutas dos veces y cambia, es
que algo no está fijado."""),

md("""## 4. Medir velocidad y memoria

Dos cifras, y las dos hay que medirlas en tu hardware antes de prometerle nada a nadie:
**tokens por segundo** y **pico de memoria**. La primera decide si la experiencia es aceptable;
la segunda, si cabe."""),
code('''import time

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
salida = modelo.generate(**entrada, max_new_tokens=200, do_sample=False)
segundos = time.perf_counter() - t0
nuevos = salida.shape[1] - entrada["input_ids"].shape[1]

print(f"{nuevos} tokens en {segundos:.1f} s")
print(f"{nuevos/segundos:.1f} tokens/s")
print(f"pico durante la generación: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"residente al terminar     : {torch.cuda.memory_allocated()/1e9:.2f} GB")
'''),
md("""**Qué deberías ver.** Del orden de **diez a dieciséis tokens por segundo** y un pico
alrededor de **6 GB**. Las cifras exactas dependen de la tarjeta que te toque y de cuánto
contexto lleves acumulado: la caché de atención crece con cada token generado, así que el pico
durante la generación es mayor que el residente tras cargar.

**Diez tokens por segundo es lento para un chat** —una respuesta de 200 tokens tarda veinte
segundos—, y esa es justo la conclusión del capítulo: un 7B cuantizado en una T4 gratuita
**sirve para aprender y para prototipar, no para dar servicio**. Para eso está el capítulo 9."""),

md("""## 5. La otra vía: GGUF y llama.cpp

`bitsandbytes` cuantiza al cargar, dentro de Python. **GGUF** es un formato que ya viene
cuantizado y lo ejecuta `llama.cpp`, un motor en C++ que corre en CPU, en Mac con Apple Silicon
y en tarjetas pequeñas, sin PyTorch de por medio.

Es un proceso de línea de comandos, así que no se lanza desde un cuaderno."""),
code('''print("""# En tu máquina, no en este cuaderno:
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8080

# Expone una API compatible con la de OpenAI en http://localhost:8080/v1
# Q4_K_M es el equilibrio habitual entre tamaño y calidad; Q8_0 pesa el doble
# y se acerca más al modelo original.
""")
'''),
md("""**Cuándo elegir cada una.** `bitsandbytes` si ya estás en Python y vas a entrenar o
ajustar; **GGUF y llama.cpp** si vas a desplegar en una máquina sin GPU decente, en un Mac o en
un portátil de alguien. En Mac, además, `bitsandbytes` tiene soporte limitado: ahí la vía es
`llama.cpp` o MLX."""),

md("""## 6. Qué anotar antes de decidir

Para cada configuración candidata, **tres cifras y una comprobación**:

| | |
|---|---|
| memoria ocupada | ¿cabe, con margen para el contexto largo? |
| tokens por segundo | ¿es tolerable para quien va a usarlo? |
| calidad | sobre 20-50 preguntas reales, no sobre una |

Y la comprobación que casi nadie hace: **que las respuestas sigan siendo correctas**. Una
cuantización agresiva puede bajar la memoria a la mitad y empezar a inventarse cifras, y eso no
sale en ninguna de las tres columnas de arriba. El capítulo 8 enseña a medirlo."""),
]

json.dump(cuaderno(celdas), open("inferencia-local-y-cuantizacion.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("escrito inferencia-local-y-cuantizacion.ipynb")
