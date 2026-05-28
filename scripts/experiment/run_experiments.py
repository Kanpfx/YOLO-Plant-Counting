from __future__ import annotations

import experiment_config as exp_cfg


def main():
    out_dir = exp_cfg.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    models = exp_cfg.build_model_list()
    main_models = exp_cfg.select_models(models, "main")
    ablation_models = exp_cfg.select_models(models, "ablation")
    generalization_models = exp_cfg.select_models(models, "generalization")

    if exp_cfg.RUN_PHYSICAL:
        from physical import run_physical

        csv_path = out_dir / "physical_stats.csv"
        run_physical(models, csv_path)
        if exp_cfg.SAVE_PLOTS:
            from plotting import plot_physical

            plot_physical(csv_path, out_dir / "physical_comparison.png", main_models)
        print(f"physical -> {csv_path}")

    if exp_cfg.RUN_PERFORMANCE:
        from performance import run_performance

        csv_path = out_dir / "performance_stats.csv"
        run_performance(models, exp_cfg.TEST_ROOT, csv_path, batch_size=exp_cfg.BATCH_SIZE)
        if exp_cfg.SAVE_PLOTS:
            from plotting import plot_performance

            plot_performance(csv_path, out_dir / "effect_comparison.png", main_models)
            plot_performance(csv_path, out_dir / "ablation_comparison.png", ablation_models)
        print(f"performance -> {csv_path}")

    if exp_cfg.RUN_GENERALIZATION:
        from generalization import run_generalization

        csv_path = out_dir / "generalization_stats.csv"
        run_generalization(
            generalization_models,
            exp_cfg.TEST_ROOT,
            csv_path,
            batch_size=exp_cfg.BATCH_SIZE,
            scales=exp_cfg.GENERALIZATION_SCALES,
        )
        if exp_cfg.SAVE_PLOTS:
            from plotting import plot_generalization

            plot_generalization(csv_path, out_dir / "generalization_comparison.png", generalization_models)
        print(f"generalization -> {csv_path}")


if __name__ == "__main__":
    main()
