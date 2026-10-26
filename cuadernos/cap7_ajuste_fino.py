#!/usr/bin/env python3
"""Cuaderno del capítulo 7: ajuste fino con LoRA en la T4 gratuita."""
import json, sys
sys.path.insert(0, ".")
from _cabecera import portada, md, code, cuaderno

celdas = portada(
    "Ajuste fino eficiente con LoRA",
    "Capítulo 7",
    "Enseñas a un modelo un **comportamiento** —tono, formato, vocabulario de tu "
    "dominio— sin tocar sus miles de millones de pesos: congelas el modelo y entrenas "
    "dos matrices pequeñas en cada capa de atención. Al final subes un adaptador de "
    "unos pocos megabytes, no un modelo entero.",
    "**GPU obligatoria.** Vale la T4 gratuita de Kaggle o de Colab: el camino de este "
    "cuaderno está medido ahí (4,15 GB de pico, 6,26 GB con *gradient checkpointing*).")

celdas += [
md("""## Antes de empezar: por qué aquí no usamos QLoRA

El libro explica QLoRA —cargar el modelo congelado en 4 bits— y este cuaderno **no lo usa**.
No es una simplificación: **hoy no funciona en una T4**.

Desde la versión 1.12, TRL convierte los adaptadores LoRA a `bfloat16` en cuanto el modelo
está cuantizado, sin comprobar si la GPU admite ese formato. La T4 no lo admite: es anterior
a la arquitectura Ampere. El error aparece en el primer paso de entrenamiento:

```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
not implemented for 'BFloat16'
```

Comprobado en octubre de 2026 con trl 1.15: falla con el código del libro, falla poniendo el
modelo base en float16, falla forzando los adaptadores a float32 —el entrenador los vuelve a
convertir— y falla con `bf16=True`, que choca contra el hardware. Tampoco se arregla bajando
a trl 1.12, 1.13 o 1.14.

**Compruébalo tú**: si algún día ejecutas la celda de entrenamiento cargando el modelo en 4
bits y arranca sin ese error, está arreglado y puedes usar QLoRA. Si sale el error, sigue
vigente lo que cuenta este recuadro.

Lo que aprendes es idéntico: **los adaptadores de bajo rango no cambian porque el modelo esté
o no cuantizado**."""),

md("""## 1. Instalación

Versiones fijadas: estas librerías rompen su propia API cada pocos meses y el código de abajo
está comprobado con estas.

**Y una línea que parece de más y no lo es.** La imagen de Kaggle trae `torchao 0.10`, y
`peft 0.21` exige más de `0.16`: en cuanto PEFT intenta averiguar si torchao está disponible,
lanza `ImportError: Found an incompatible version of torchao` y el cuaderno se para antes de
entrenar. Aquí no se usa torchao para nada, así que se desinstala y PEFT sigue su camino.
Comprobado en octubre de 2026; si algún día la imagen trae una versión más nueva, esta línea
sobra y no estorba."""),
code("""%pip uninstall -q -y torchao
%pip install -q "transformers==5.19.0" "trl==1.15.0" "peft==0.21.2" \\
    "datasets==5.1.0" "accelerate" "sentencepiece"
"""),
code("""import os
# Kaggle ofrece «GPU T4 x2» y, con dos GPU visibles, Trainer envuelve el modelo en
# nn.DataParallel — que SFTTrainer NO admite: «SFTTrainer does not support
# nn.DataParallel». Con una sola visible entrena sin tocar nada más. Esta línea va
# ANTES de importar torch, o CUDA ya se ha inicializado con las dos.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch, transformers, trl, peft
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna")
print("GPU visibles:", torch.cuda.device_count())
print("bf16 nativo:", torch.cuda.is_bf16_supported(including_emulation=False))
print("transformers", transformers.__version__, "· trl", trl.__version__, "· peft", peft.__version__)
"""),
md("""**Qué deberías ver.** `Tesla T4`, **`GPU visibles: 1`**, `bf16 nativo: False` y las tres
versiones de arriba. Ese `False` es exactamente la razón del recuadro anterior, y ese `1` la
de la línea de `CUDA_VISIBLE_DEVICES`."""),

md("""## 2. El dataset

El formato estándar del ajuste con ejemplos es una lista de conversaciones. **Doscientos
ejemplos bien escritos valen más que diez mil ruidosos.**

Fuentes habituales: históricos de tickets resueltos y anonimizados, respuestas modelo del
manual de atención al cliente, o ejemplos sintéticos generados con un modelo grande y
revisados por una persona que sepa del tema.

Aquí se genera un dataset de juguete con la forma correcta, para que el cuaderno corra sin
descargar nada. Sustitúyelo por el tuyo cuando lo tengas."""),
code('''import json, random
from datasets import load_dataset

random.seed(42)
SISTEMA = "Eres el asistente de atención al socio de una cooperativa eléctrica. Respondes en español, en dos o tres frases, y pides el número de contrato cuando hace falta."
CASOS = [
    ("Me han cobrado dos veces la factura de agosto",
     "Lamento la incidencia. Para revisarlo necesito el número de contrato que aparece arriba a la derecha de la factura."),
    ("¿Cuándo entra en vigor la nueva tarifa?",
     "La tarifa revisada se aplica desde el primer día del mes siguiente a su aprobación en asamblea. Te confirmo la fecha exacta con tu número de contrato."),
    ("Quiero darme de baja",
     "Puedo tramitarlo. Necesito tu número de contrato y la fecha en la que quieres que sea efectiva la baja."),
]
with open("sft.jsonl", "w", encoding="utf-8") as f:
    for _ in range(200):
        pregunta, respuesta = random.choice(CASOS)
        f.write(json.dumps({"messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": pregunta},
            {"role": "assistant", "content": respuesta}]}, ensure_ascii=False) + "\\n")

ds = load_dataset("json", data_files="sft.jsonl")["train"].train_test_split(test_size=0.1, seed=42)
print(ds)
print(ds["train"][0]["messages"][1])
'''),
md("""**Qué deberías ver.** Un `DatasetDict` con 180 ejemplos de entrenamiento y 20 de
validación, y la primera pregunta del usuario impresa debajo."""),

md("""## 3. El modelo: por qué este y no uno más grande

**Qwen2.5-1.5B-Instruct**, en 16 bits y sin cuantizar. Dos razones:

- **Cabe de sobra en una T4.** Medido: 4,15 GB de pico durante el entrenamiento, 6,26 GB con
  *gradient checkpointing*.
- **Su licencia es Apache 2.0**, así que el asistente se puede usar comercialmente.

Cuidado con subir de tamaño por subir: **Qwen2.5-3B se publica bajo la Qwen Research License,
que no permite uso comercial.** Si el asistente es para una empresa, ese modelo no sirve
aunque quepa."""),
code('''import torch
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODELO)
# Sin cuantizar: así TRL no convierte los adaptadores a bfloat16 y entrena en la T4
modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float16).to("cuda")
print(f"{sum(p.numel() for p in modelo.parameters())/1e9:.2f} B parámetros")
'''),

md("""## 4. LoRA: qué entrena y qué no

En lugar de modificar los miles de millones de pesos, LoRA **congela el modelo** y entrena dos
matrices pequeñas, A y B, en cada capa de atención. La actualización de una matriz de pesos W
se aproxima como:

$$W' = W + \\frac{\\alpha}{r}\\,B A,\\quad B \\in \\mathbb{R}^{d \\times r},\\ A \\in \\mathbb{R}^{r \\times k},\\ r \\ll \\min(d,k)$$

El **rango** `r` controla cuánto puede aprender el adaptador. Con `r = 16` se entrena menos del
1 % de los parámetros, y por eso el adaptador ocupa unos pocos megabytes y se distribuye por
separado del modelo base.

`lora_alpha` escala la actualización; la convención habitual es el doble del rango.
`target_modules` elige en qué proyecciones de la atención entra."""),
code('''lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
print(lora)
'''),
md("""**Qué deberías ver.** La configuración impresa. El recuento de lo que se entrena de
verdad lo imprime el entrenador en la celda siguiente, y es la idea entera del capítulo."""),

md("""## 5. Entrenar

`bf16` y `fp16` se eligen **mirando la GPU**, no a mano: en la T4 el primero es `False` y se
entrena en `fp16`. En una GPU Ampere o superior sale al revés y no hay que tocar nada.

`gradient_accumulation_steps` suma varios lotes pequeños antes de actualizar, para simular un
lote grande sin quedarse sin memoria: aquí 2 × 8 = 16 ejemplos por actualización."""),
code('''bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
config = SFTConfig(
    output_dir="asistente-cooperativa-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",
    bf16=bf16,
    fp16=not bf16,
    report_to=[],
)
trainer = SFTTrainer(
    model=modelo,              # en 16 bits, sin cuantizar (ver el recuadro del principio)
    args=config,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    peft_config=lora,
)
trainer.model.print_trainable_parameters()     # menos del 1 % del modelo
trainer.train()
trainer.save_model()           # guarda SOLO el adaptador: unos pocos MB
print("pico de memoria:", round(torch.cuda.max_memory_allocated()/1e9, 2), "GB")
'''),
md("""**Qué deberías ver.** Tres épocas, la pérdida de entrenamiento bajando y la de validación
al final de cada una. Y un pico de memoria en torno a **3,2 GB** con este dataset de juguete —el libro mide
4,15 GB con el suyo, que tiene secuencias más largas—: por eso cabe.

**Si la pérdida de validación sube mientras la de entrenamiento baja**, es sobreajuste: menos
épocas, más datos o más `lora_dropout`. **Si la pérdida no baja**, suele ser la tasa de
aprendizaje o la plantilla de chat: prueba entre 1e-4 y 3e-4 y revisa el formato del dataset."""),

md("""## 6. Probar el adaptador antes de publicar nada"""),
code('''from peft import PeftModel

afinado = PeftModel.from_pretrained(modelo, "asistente-cooperativa-lora")
mensajes = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": "Me han cobrado dos veces la factura de agosto"}]
# En transformers 5, `apply_chat_template` con `return_dict=True` devuelve un
# diccionario con `input_ids` y `attention_mask`, no un tensor suelto. Pasarle el
# tensor a `generate()` da un AttributeError seco, sin mensaje: es uno de los
# cambios de API de los que avisa el prólogo del libro.
entrada = tok.apply_chat_template(mensajes, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
salida = afinado.generate(**entrada, max_new_tokens=80, do_sample=False)
print(tok.decode(salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True))
'''),
md("""**Qué deberías ver.** Una respuesta corta, en español, pidiendo el número de contrato: el
comportamiento que enseñaba el dataset. Con 200 ejemplos de juguete será repetitiva — es lo
esperable, y es justo lo que mide el capítulo 8."""),

md("""## 7. Fusionar y publicar

Fusionar sobre un modelo cuantizado produciría pesos degradados, así que el base se recarga en
16 bits. Con el 1.5B son unos 3 GB y la fusión sale en la misma T4 donde acabas de entrenar;
si el día de mañana ajustas un 7B en una GPU Ampere, cuenta unos 15 GB y hazla en CPU con
memoria alta o en otra máquina.

**Publícalo como privado hasta haber superado la evaluación del capítulo 8.**"""),
code('''# Descomenta cuando tengas cuenta en el Hub y el adaptador evaluado
# from huggingface_hub import notebook_login; notebook_login()
#
# base = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float16)
# fusionado = PeftModel.from_pretrained(base, "asistente-cooperativa-lora").merge_and_unload()
# fusionado.push_to_hub("tu-usuario/asistente-cooperativa", private=True)
# tok.push_to_hub("tu-usuario/asistente-cooperativa", private=True)
'''),

md("""## 7b. ¿Y el *gradient checkpointing*? Mídelo, no lo supongas

El *gradient checkpointing* **cambia cómputo por memoria**: en vez de guardar las activaciones
de cada capa para la retropropagación, las recalcula cuando hacen falta. Entrena más despacio y
ocupa menos.

Esa es la teoría, y es la razón por la que la tabla de errores típicos del capítulo lo
recomienda para resolver un `CUDA OOM`. La celda siguiente lo comprueba en los dos modos, con
unos pocos pasos para que sea rápido."""),
code('''import gc

def pico(checkpointing, pasos=20):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float16).to("cuda")
    cfg = SFTConfig(output_dir=f"tmp-{checkpointing}", max_steps=pasos,
                    per_device_train_batch_size=2, gradient_accumulation_steps=1,
                    learning_rate=2e-4, logging_steps=pasos, report_to=[],
                    bf16=bf16, fp16=not bf16,
                    gradient_checkpointing=checkpointing)
    t = SFTTrainer(model=m, args=cfg, train_dataset=ds["train"], peft_config=lora)
    t0 = time.perf_counter(); t.train(); seg = time.perf_counter() - t0
    gb = torch.cuda.max_memory_allocated() / 1e9
    del t, m; gc.collect(); torch.cuda.empty_cache()
    return gb, seg

import time
sin_, t_sin = pico(False)
con_, t_con = pico(True)
print(f"sin checkpointing: {sin_:.2f} GB · {t_sin:.1f} s")
print(f"con checkpointing: {con_:.2f} GB · {t_con:.1f} s")
print(f"memoria: {(con_-sin_)/sin_*100:+.0f} %   ·   tiempo: {(t_con-t_sin)/t_sin*100:+.0f} %")
'''),
md("""**Qué deberías ver, y es una predicción que la celda puede desmentir.** Si el mecanismo es
el que se describe, **la memoria tiene que bajar y el tiempo subir**. Si te sale al revés,
no es que el *checkpointing* haya cambiado: es que algo más cambió entre las dos medidas —el
tamaño de lote, la longitud de las secuencias o la memoria que quedaba ocupada de antes—.

Por eso las dos medidas de esta celda **liberan la memoria y reinician el contador** antes de
cada una. Comparar dos picos medidos sin reiniciar es la forma más fácil de concluir lo
contrario de lo que pasa."""),

md("""## 8. Más allá del ajuste con ejemplos

Cuando tengas pares de respuestas «preferida» y «rechazada» —por ejemplo, valoraciones de
agentes humanos—, TRL permite aplicar **DPO** con `DPOTrainer`, que alinea el modelo con las
preferencias sin entrenar un modelo de recompensa aparte.

**Y la advertencia del principio vale igual aquí, aunque cambies de entrenador:** el casting a
bfloat16 no es cosa del `SFTTrainer`, está en los siete entrenadores de TRL —SFT, DPO, GRPO,
KTO, Reward, RLOO y destilación—, así que `DPOTrainer` sobre un modelo cuantizado falla en la
T4 con el mismo error. Sin cuantizar, funciona."""),
]

json.dump(cuaderno(celdas), open("cap7-ajuste-fino-con-lora.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("escrito cap7-ajuste-fino-con-lora.ipynb")
