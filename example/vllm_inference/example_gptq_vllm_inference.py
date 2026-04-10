"""

Example: Quantize a model with OneComp and run inference with vLLM

Performs the following steps:
  1. Quantize with GPTQ (4-bit, groupsize=128)
  2. Save the quantized model
  3. Load the quantized model with vLLM's offline LLM interface
  4. Generate text

Requirements:
  pip install vllm

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import gc

import torch
from onecomp import Runner, ModelConfig, CalibrationConfig, GPTQ, setup_logger
from vllm import LLM, SamplingParams


def main():
    setup_logger()

    # Step 1: Quantize with GPTQ
    save_dir = "./TinyLlama-1.1B-Chat-gptq-4bit"

    model_config = ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    quantizer = GPTQ(wbits=4, groupsize=128)
    calibration_config = CalibrationConfig(
        num_calibration_samples=128,
        max_length=512,
    )
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calibration_config,
        qep=False,
    )
    runner.run()

    # Step 2: Save the quantized model
    runner.save_quantized_model(save_dir)

    # Free GPU memory used by quantization before loading vLLM
    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Load the quantized model with vLLM
    llm = LLM(
        model=save_dir,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
    )

    # Step 4: Generate text
    prompts = [
        "Explain what post-training quantization is in one sentence:",
        "The capital of France is",
    ]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.0))

    for output in outputs:
        print(f"Prompt:   {output.prompt}")
        print(f"Response: {output.outputs[0].text}")
        print()


if __name__ == "__main__":
    main()
