# Los cuadernos del libro

Están en **Kaggle**, no en Colab: se abren y se ejecutan sin instalar nada y, a
diferencia de Colab, el enlace no depende de una cuenta de Google Drive. El cuaderno
del capítulo 7 necesita GPU, que Kaggle da gratis con una cuota semanal.

| Cuaderno | Capítulos | Fuente en este repositorio | GPU |
|---|---|---|---|
| [kobalto-hf-fundamentos](https://www.kaggle.com/code/alesrin/kobalto-hf-fundamentos) | 1, 2 y 3 | `fundamentos.py` | no |
| [rag-conectar-la-ia-con-tus-documentos](https://www.kaggle.com/code/alesrin/rag-conectar-la-ia-con-tus-documentos) | 6 y 8 | `rag.py` | no |
| [inferencia-local-y-cuantizacion](https://www.kaggle.com/code/alesrin/inferencia-local-y-cuantizacion) | 4 y 5 | `inferencia.py` | no |
| [ajuste-fino-con-lora](https://www.kaggle.com/code/alesrin/ajuste-fino-con-lora) | 7 | `cap7_ajuste_fino.py` | sí |
| [agentes-mcp-y-despliegue](https://www.kaggle.com/code/alesrin/agentes-mcp-y-despliegue) | 9 y 15 | `agentes_mcp.py` | no |
| [crea-tu-propia-ia-con-hugging-face-18-ejemplos](https://www.kaggle.com/code/alesrin/crea-tu-propia-ia-con-hugging-face-18-ejemplos) | 12 | — (se edita a mano) | no |

Los cinco primeros **no se editan a mano**: se escriben en su `.py` y se regeneran con

    python herramientas/generar_cuadernos.py

El `.py` solo monta celdas; no ejecuta modelos. La generación es determinista, así que
`--verifica` compara byte a byte y la integración continua caza cualquier cuaderno
editado por fuera, que es un cambio que se perdería en la siguiente generación.

## Los seis PDF de la cooperativa y el cuaderno de RAG

El cuaderno del capítulo 6 **genera tres PDF de juguete** en lugar de usar los seis del
dataset, para que corra sin descargar nada; lo dice en su segunda celda. Los seis de
verdad están en este repositorio, en `datos/cooperativa/documentos/`, y son los que dan
las cifras del libro: 43 fragmentos y Recall@k del apartado 8.3.
