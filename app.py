#!/usr/bin/env python3
"""Capítulo 9: la interfaz del asistente, para un Space privado.

    python app.py

Antes de abrirlo a usuarios reales, la lista de comprobación del apartado 9.4:
autenticación delante del endpoint, límite de peticiones por usuario, registro con
anonimización, monitorización, botón de valoración y plan de reversión por hash.

El `HF_TOKEN` va en los *secrets* del Space, nunca en el código.
"""
import os

MODELO = os.environ.get("MODELO_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTADOR = os.environ.get("ADAPTADOR_LORA")   # la carpeta que deja entrenar.py


def construye():
    import gradio as gr
    from transformers import pipeline

    from recuperador import Recuperador, contexto_y_mensajes

    rec = Recuperador()
    chat = pipeline("text-generation", model=MODELO, device_map="auto")
    if ADAPTADOR:
        chat.model.load_adapter(ADAPTADOR)

    def chatear(mensaje, historial):
        recuperados = rec.busca(mensaje, k=4)
        mensajes = contexto_y_mensajes(mensaje, recuperados)
        respuesta = chat(mensajes, max_new_tokens=300, do_sample=False)[0]["generated_text"][-1]
        fuentes = ", ".join(f"[{i + 1}] {m['documento']} p. {m['pagina']}"
                            for i, (_, m, _) in enumerate(recuperados))
        return f"{respuesta['content'].strip()}\n\nFuentes: {fuentes}"

    return gr.ChatInterface(
        chatear,
        title="Asistente de la cooperativa",
        description="Responde solo con los documentos de la cooperativa y cita la fuente.",
        examples=["¿Cómo cambio la potencia contratada?", "¿Qué es el bono social?"],
    )


if __name__ == "__main__":
    construye().launch()
