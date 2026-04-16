"""JointQ Benchmark

Run JointQ for all combinations of bits × group_size in a single pass.
Shares calibration data accumulation across quantizers for efficiency.
Results are saved under output_dir.

Copyright 2025-2026 Fujitsu Ltd.

Usage:
    python quant_benchmark.py
"""

import itertools

import hydra
from omegaconf import DictConfig, OmegaConf

from onecomp import CalibrationConfig, JointQ, ModelConfig, Runner


def create_quantizers(cfg: DictConfig):
    """Create a list of JointQ quantizers for all combinations of bits × group_size."""
    quantizers = []
    sym = cfg.jointq.symmetric
    sym_label = "sym" if sym else "asym"

    for bits, gs in itertools.product(cfg.jointq.bits, cfg.jointq.group_size):
        # Label strings
        gs_label = "pc" if gs is None else f"gs{gs}"

        # JointQ: group_size=None means per-channel
        jointq_groupsize = None if gs is None else gs

        # Build ILS kwargs
        ils_kwargs = {}
        if cfg.jointq.ils_enabled:
            ils_kwargs = {
                "ils_enabled": True,
                "ils_num_iterations": cfg.jointq.ils_num_iterations,
                "ils_num_clones": cfg.jointq.ils_num_clones,
                "ils_num_channels": cfg.jointq.ils_num_channels,
            }

        quantizers.append(
            JointQ(
                num_layers=cfg.jointq.num_layers,
                bits=bits,
                symmetric=sym,
                group_size=jointq_groupsize,
                batch_size=cfg.jointq.batch_size,
                log_level=cfg.jointq.log_level,
                device=cfg.jointq.device,
                regularization_lambda=cfg.jointq.regularization_lambda,
                actorder=cfg.jointq.actorder,
                calc_quant_error=True,
                name=f"JointQ_{bits}bit_{gs_label}_{sym_label}",
                **ils_kwargs,
            )
        )

    return quantizers


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_qwen3-14b")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model_config = ModelConfig(path=cfg.model_path, device=cfg.model_device)

    quantizers = create_quantizers(cfg)

    print(f"Number of quantizers: {len(quantizers)}")
    for q in quantizers:
        print(f"  - {q.name}")

    # Build Runner
    runner = Runner(
        model_config=model_config,
        quantizers=quantizers,
        calibration_config=CalibrationConfig(
            max_length=cfg.max_length,
            num_calibration_samples=cfg.num_calibration_samples,
            strategy=cfg.calibration_strategy,
            seed=cfg.calibration_seed,
            batch_size=cfg.calibration_batch_size,
        ),
    )

    # Run quantization
    runner.run()

    # Save results
    for q in quantizers:
        runner.save_quantization_statistics(
            f"quantization_statistics_{q.name}.json", quantizer=q
        )

    # Perplexity evaluation
    if cfg.calc_ppl:
        runner.benchmark_perplexity(original_model=cfg.calc_original_ppl)

    # Accuracy evaluation
    if cfg.calc_acc:
        runner.benchmark_accuracy(original_model=cfg.calc_original_acc)


if __name__ == "__main__":
    main()
