#!/usr/bin/env python3
"""Capítulo 7: ajustar el tono y el formato de la cooperativa con LoRA.

    python entrenar.py

Entrena un adaptador de bajo rango sobre `Qwen/Qwen2.5-1.5B-Instruct` con las 500
conversaciones de `datos/cooperativa/sft.jsonl`. El adaptador pesa unos pocos
megabytes y se distribuye aparte del modelo base.

**El modelo NO se carga cuantizado, y es una decisión, no un descuido.** Desde la
versión 1.12, TRL convierte los adaptadores LoRA a bfloat16 en cuanto el modelo está
cuantizado, sin comprobar si la GPU admite ese formato; una T4 no lo admite y el
entrenamiento muere en el primer paso con
`NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented
for 'BFloat16'`. Con el modelo en 16 bits, LoRA entrena igual y lo que se aprende es
idéntico. En una GPU Ampere o superior se puede añadir `--cuatro-bits` y hacer QLoRA.

Comprobado en octubre de 2026 con trl 1.15, peft 0.21 y transformers 5.19. Si ejecuta
`--cuatro-bits` en una T4 y el entrenamiento arranca, el fallo de TRL está resuelto.
"""
import argparse
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.join(RAIZ, "datos", "cooperativa", "sft.jsonl")
BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # Apache 2.0: uso comercial permitido


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", default=BASE)
    ap.add_argument("--salida", default=os.path.join(RAIZ, "asistente-cooperativa-lora"))
    ap.add_argument("--epocas", type=float, default=3)
    ap.add_argument("--cuatro-bits", action="store_true",
                    help="QLoRA: solo en GPU Ampere o superior (ver el aviso de arriba)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=SFT)["train"]
    ds = ds.train_test_split(test_size=0.1, seed=42)
    print(ds)          # 450 ejemplos de entrenamiento y 50 de validación

    carga = {"dtype": torch.float16}
    if args.cuatro_bits:
        from transformers import BitsAndBytesConfig
        carga["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    modelo = AutoModelForCausalLM.from_pretrained(args.modelo, **carga)
    if not args.cuatro_bits:
        modelo = modelo.to("cuda" if torch.cuda.is_available() else "cpu")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.salida,
        num_train_epochs=args.epocas,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        bf16=bf16,
        fp16=not bf16,
    )
    trainer = SFTTrainer(model=modelo, args=config, train_dataset=ds["train"],
                         eval_dataset=ds["test"], peft_config=lora)
    trainer.train()
    trainer.save_model()          # guarda solo el adaptador LoRA
    print(f"adaptador guardado en {os.path.relpath(args.salida, RAIZ)}/")


if __name__ == "__main__":
    main()
