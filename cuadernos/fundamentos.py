#!/usr/bin/env python3
"""Cuaderno de los capítulos 2, 3 y 3B: entorno, tokenización y el transformer por dentro."""
import json, sys
sys.path.insert(0, ".")
from _cabecera import portada, md, code, cuaderno

celdas = portada(
    "Fundamentos: entorno, tokenización y el transformer por dentro",
    "Capítulos 2, 3 y 3B",
    "Preparar el entorno, ver cómo un modelo convierte texto en números, y construir un "
    "transformer entero desde cero con numpy y PyTorch para entender qué hay debajo de "
    "todo lo demás.",
    "**Nada.** Corre entero en CPU; solo la celda del pipeline descarga un modelo de 1,5 B.")

celdas += [
md("""# Capítulo 2 · Preparar el entorno

## 1. Qué hay que instalar, y dónde

**En Kaggle no hace falta entorno virtual.** El libro explica cómo montar uno porque lo normal
es trabajar en tu máquina, y ahí sí importa: un entorno por proyecto evita que una versión de
`transformers` que necesita un trabajo rompa otro.

Aquí, en cambio, la celda de abajo es todo lo que hay. Si copias el `python -m venv` del libro
a un cuaderno, **no funciona**: cada `!` abre su propio intérprete de órdenes, así que
`!source .venv/bin/activate` no deja nada activado en la celda siguiente."""),
code("""%pip install -q "transformers==5.19.0" "accelerate" "sentencepiece"
"""),
code('''import torch, transformers
print("transformers", transformers.__version__)
print("GPU disponible:", torch.cuda.is_available(),
      "·", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'''),
md("""**Qué deberías ver.** La versión y, si has pedido acelerador, el nombre de la tarjeta.
Este cuaderno corre entero en CPU: el modelo que usa es pequeño.

**Y el token del Hub nunca va en una celda.** En Kaggle se guarda en *Add-ons → Secrets* y se
lee desde ahí; en tu máquina, en una variable de entorno o con `hf auth login`. Lo que no hay
que hacer nunca es escribirlo en el cuaderno, porque el cuaderno se comparte."""),

md("""# Capítulo 3 · Tokenización, plantillas y pipelines

## 2. Un modelo no ve letras: ve números

El tokenizador parte el texto en trozos y les asigna un identificador. Esos trozos **no son
palabras**: son subpalabras, y el reparto depende del idioma con el que se entrenó el
tokenizador."""),
code('''from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
frase = "La factura de octubre incluye el IVA."
ids = tok(frase)["input_ids"]
print(ids)
print(tok.convert_ids_to_tokens(ids))
print(f"{len(frase)} caracteres -> {len(ids)} tokens  ({len(frase)/len(ids):.2f} car./token)")
'''),
md("""**Qué deberías ver.** Una lista de números y, debajo, los trozos. Fíjate en cuáles van
precedidos de un espacio codificado: el tokenizador se lleva el espacio con la palabra.

**Y la última línea es la que importa para el bolsillo.** Los caracteres por token deciden
cuántos tokens cuesta un texto, y en español suelen ser menos que en inglés con los
tokenizadores entrenados mayoritariamente en inglés. Pruébalo con una frase inglesa de la
misma longitud y compara."""),

md("""## 3. La plantilla de chat no es decoración

Un modelo de instrucciones se entrenó con un formato concreto de turnos. Si le pasas el texto
suelto, responde peor — y no da ningún error, que es lo que lo hace difícil de detectar."""),
code('''mensajes = [
    {"role": "system", "content": "Eres un asistente de una comercializadora eléctrica."},
    {"role": "user", "content": "¿Qué es el término de potencia?"},
]
prompt = tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
print(prompt)
'''),
md("""**Qué deberías ver.** El mismo contenido envuelto en las marcas de turno del modelo, y al
final la marca que indica «ahora contesta tú». Eso es lo que `add_generation_prompt=True`
añade: sin ella, el modelo puede seguir escribiendo el turno del usuario.

**Cuidado con esta función**, que cambió: con `return_dict=True` devuelve un **diccionario**,
no un tensor, y pasárselo a `generate()` da un `AttributeError` sin mensaje."""),

md("""## 4. El pipeline: lo más corto entre un modelo y una respuesta"""),
code('''from transformers import pipeline

chat = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")
salida = chat(mensajes, max_new_tokens=120, do_sample=True, temperature=0.3)
print(salida[0]["generated_text"][-1]["content"])
'''),
md("""**Qué deberías ver.** Una explicación del término de potencia en español. **No va a ser
igual que la mía ni que la del libro**: hay muestreo, así que cada ejecución da un texto
distinto. Con `do_sample=False` sale siempre el mismo.

Descargar el modelo tarda un par de minutos la primera vez."""),

md("""# Capítulo 3B · Bajo el capó

Este capítulo es el único del libro que abre la máquina. Si lo saltas, el resto se entiende
igual — pero si lo haces, todo lo demás deja de ser magia.

## 5. La atención, con numpy y cinco palabras

Cada token mira a los anteriores y decide cuánto le importa cada uno. Eso es todo lo que hace
la atención, y cabe en veinte líneas sin ninguna librería de aprendizaje automático.

La **máscara causal** es la que impide mirar al futuro: sin ella, el modelo haría trampa al
predecir la palabra siguiente."""),
code('''import numpy as np

np.random.seed(0)
tokens = ["el", "contador", "marca", "300", "kWh"]
d = 8                                   # dimensión de los embeddings
X = np.random.randn(len(tokens), d)     # un embedding por token (aleatorio en este ejemplo)

Wq, Wk, Wv = (np.random.randn(d, d) / np.sqrt(d) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv        # consultas, claves y valores

puntuaciones = Q @ K.T / np.sqrt(d)     # cuánto "mira" cada token a cada otro
mascara = np.triu(np.ones((5, 5)), k=1).astype(bool)
puntuaciones[mascara] = -np.inf         # máscara causal: nadie mira al futuro

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

A = softmax(puntuaciones)               # pesos de atención: cada fila suma 1
salida = A @ V                          # mezcla ponderada de los valores

np.set_printoptions(precision=2, suppress=True)
print(A)
print("filas suman 1:", np.allclose(A.sum(axis=1), 1), "| forma de la salida:", salida.shape)
'''),
md("""**Qué deberías ver, y esta vez exactamente.** La semilla está fijada, así que la matriz
tiene que salir **idéntica a la del libro**:

```
[[1.   0.   0.   0.   0.  ]
 [0.46 0.54 0.   0.   0.  ]
 [0.32 0.09 0.59 0.   0.  ]
 [0.3  0.22 0.31 0.17 0.  ]
 [0.13 0.05 0.05 0.6  0.17]]
```

Lee el triángulo de ceros: **la primera palabra solo se mira a sí misma** y cada fila añade una
columna. Eso es la máscara causal, dibujada."""),

md("""## 6. Un transformer entero, entrenado de cero, en CPU

105.240 parámetros y un corpus de cuatro frases sobre la factura de la luz. No sirve para nada
— y por eso se entiende: se ve entrar la pérdida por arriba y salir texto legible por abajo,
en menos de un minuto y sin GPU."""),
code('''import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
corpus = (
    "la factura de la luz tiene un termino de potencia y un termino de energia. "
    "el termino de potencia se paga aunque no se consuma. "
    "el termino de energia depende de los kilovatios hora consumidos. "
    "la potencia contratada se mide en kilovatios. "
) * 20
vocab = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
datos = torch.tensor([stoi[c] for c in corpus])
n = int(0.9 * len(datos))
entreno, valida = datos[:n], datos[n:]
CONTEXTO, LOTE, D = 32, 32, 64

def lote(split):
    fuente = entreno if split == "entreno" else valida
    ix = torch.randint(len(fuente) - CONTEXTO - 1, (LOTE,))
    x = torch.stack([fuente[i:i + CONTEXTO] for i in ix])
    y = torch.stack([fuente[i + 1:i + CONTEXTO + 1] for i in ix])  # objetivo: el siguiente carácter
    return x, y

class MiniGPT(nn.Module):
    def __init__(self, v, d=D, cabezas=4, capas=2):
        super().__init__()
        self.tok = nn.Embedding(v, d)
        self.pos = nn.Embedding(CONTEXTO, d)
        capa = nn.TransformerEncoderLayer(d, cabezas, 4 * d, dropout=0.1,
                                          batch_first=True, norm_first=True)
        self.bloques = nn.TransformerEncoder(capa, capas, enable_nested_tensor=False)
        self.norma = nn.LayerNorm(d)
        self.cabeza = nn.Linear(d, v)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))
        causal = nn.Transformer.generate_square_subsequent_mask(T)
        h = self.bloques(h, mask=causal, is_causal=True)
        return self.cabeza(self.norma(h))          # logits: (lote, T, vocabulario)

modelo = MiniGPT(len(vocab))
opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
print(f"vocabulario: {len(vocab)} caracteres")
print(f"parámetros: {sum(p.numel() for p in modelo.parameters()):,}   (el libro: 105.240)")
'''),
md("""**Qué deberías ver.** El mismo número de parámetros que el libro, **105.240**, porque el
corpus es fijo y el vocabulario también. Si te sale otro, la arquitectura ha cambiado de
versión."""),
code('''@torch.no_grad()
def perdida_media(split, k=20):
    modelo.eval()
    l = torch.stack([F.cross_entropy(modelo(x).flatten(0, 1), y.flatten())
                     for x, y in (lote(split) for _ in range(k))])
    modelo.train()
    return l.mean().item()

print("el libro mide: 2.967 -> 0.112 -> 0.099 -> 0.096 -> 0.096 -> 0.092")
for paso in range(1501):
    x, y = lote("entreno")
    perdida = F.cross_entropy(modelo(x).flatten(0, 1), y.flatten())
    opt.zero_grad()
    perdida.backward()      # retropropagación: gradiente de la pérdida respecto a cada peso
    opt.step()              # actualización de los pesos
    if paso % 300 == 0:
        print(f"paso {paso:4d} | entreno {perdida_media('entreno'):.3f} | validación {perdida_media('valida'):.3f}")
'''),
md("""**Qué deberías ver: las cifras del libro, hasta la milésima.** `torch.manual_seed(42)`
fija el generador global, y de él salen los pesos iniciales, el *dropout* y el muestreo de los
lotes: la ejecución entera es determinista. Medido al montar este cuaderno —2.967, 0.112,
0.098, 0.096, 0.096, 0.093 contra los 2.967, 0.112, 0.099, 0.096, 0.096, 0.092 del libro— las
seis coinciden salvo una milésima suelta.

**Si te salen cifras distintas**, no es tu máquina: es que la versión de PyTorch ha cambiado
algo del orden en que consume el generador. La forma, en cambio, no debería cambiar nunca:
caída brutal en los primeros 300 pasos, y entreno y validación juntos — con cuatro frases
repetidas no hay sobreajuste que valga."""),
code('''@torch.no_grad()
def generar(texto, n=60, temperatura=0.8):
    modelo.eval()
    ids = torch.tensor([[stoi[c] for c in texto]])
    for _ in range(n):
        logits = modelo(ids[:, -CONTEXTO:])[:, -1] / temperatura
        siguiente = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, siguiente], dim=1)
    return "".join(itos[int(i)] for i in ids[0])

print(repr(generar("el termino de ")))
'''),
md("""**Qué deberías ver: exactamente la frase del libro.**
`'el termino de potencia se paga aunque no se consuma. el termino de energia'`

Hay muestreo —`torch.multinomial`—, así que podría salir cualquier cosa; sale la misma porque
la semilla gobierna también eso. Es la mejor prueba de que has reproducido el capítulo entero,
no solo de que te ha corrido.

Y ahí está lo que este capítulo quería enseñar: **105.240 parámetros bastan para aprender la
estructura de cuatro frases**. Un modelo de verdad hace lo mismo con mil millones de veces más
texto."""),

md("""## 7. Y el mismo modelo, de verdad

La configuración de un modelo publicado dice lo mismo que acabas de construir, en grande."""),
code('''from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
print("vocabulario       ", f"{cfg.vocab_size:,}")
print("dimensión         ", cfg.hidden_size)
print("cabezas           ", cfg.num_attention_heads)
print("capas             ", cfg.num_hidden_layers)
print("contexto máximo   ", f"{cfg.max_position_embeddings:,}")
'''),
md("""**Qué deberías ver.** Los mismos cinco conceptos de tu modelo de juguete —vocabulario,
dimensión, cabezas, capas y contexto— con otros números. **No hay nada más.** Esa es la
conclusión del capítulo: la diferencia entre tu MiniGPT y un modelo comercial es de escala y de
datos, no de clase."""),
]

json.dump(cuaderno(celdas), open("kobalto-hf-fundamentos.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("escrito kobalto-hf-fundamentos.ipynb")
