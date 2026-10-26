# Licencias

Este repositorio tiene tres materiales con tres licencias distintas. El reparto y el
motivo de cada elección son los que publica el anexo B del libro, y se citan aquí tal
como aparecen allí:

| Material | Licencia | Motivo (anexo B del libro) |
|---|---|---|
| Texto del libro y explicaciones de los cuadernos | CC BY-SA 4.0 | «Cualquiera puede adaptarlo citando la fuente, y los materiales didácticos derivados siguen siendo abiertos» |
| Dataset de la cooperativa | CC BY 4.0 | «Es la licencia habitual de los datasets del Hub y evita la duda de si un modelo entrenado con los datos debe heredar la licencia» |
| Código | Apache 2.0 | «Permite reutilizarlo en proyectos profesionales, también comerciales, conservando los avisos» |

## Qué fichero cae en cada licencia

* **Apache 2.0** — todo el código: `indexar.py`, `entrenar.py`, `evaluar.py`,
  `recuperador.py`, `app.py`, `herramientas/`, `tests/` y los generadores
  `datos/cooperativa/genera_pdf.py`, `genera_sft.py` y `linea_base_bm25.py`.
* **CC BY 4.0** — el dataset: `datos/cooperativa/documentos/` (los seis PDF y su
  texto fuente), `semillas.json`, `sft.jsonl` y `eval.jsonl`.
* **CC BY-SA 4.0** — los textos explicativos: `README.md`, `docs/` y las
  explicaciones en prosa de los cuadernos.

Atribución: Kobalto Educación, «Crea tu propia IA» (2026).

Los seis documentos de la cooperativa son **ficticios**. La Cooperativa Eléctrica
Valle Verde no existe, y ni sus precios, ni sus plazos, ni sus condiciones
corresponden a una entidad real o a la normativa vigente: están escritos para que un
sistema RAG tenga algo verificable que recuperar.

## Textos legales completos

Los tres están en `licencias/`, descargados sin modificar de quien los publica:

| Fichero | Origen | Descargado | Tamaño |
|---|---|---|---|
| `licencias/apache-2.0.txt` | https://www.apache.org/licenses/LICENSE-2.0.txt | 26-10-2026 | 11.358 bytes |
| `licencias/cc-by-4.0.txt` | https://creativecommons.org/licenses/by/4.0/legalcode.txt | 26-10-2026 | 18.657 bytes |
| `licencias/cc-by-sa-4.0.txt` | https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt | 26-10-2026 | 20.138 bytes |

Los ficheros están **sin tocar**, con el contenido exacto que sirven esas direcciones,
para que se puedan comparar byte a byte con el original. Creative Commons publica el
texto en formato plano en inglés; las traducciones al español que publica en su propio
sitio están en
https://creativecommons.org/licenses/by/4.0/legalcode.es y
https://creativecommons.org/licenses/by-sa/4.0/legalcode.es.

Cualquier duda sobre qué permite cada licencia se resuelve en su texto, no en este
fichero ni en el libro: lo de arriba es el reparto de materiales, no una
interpretación de las licencias.
