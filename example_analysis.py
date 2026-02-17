#!/usr/bin/env python3
"""Example script demonstrating autotune_attention and KernelAnalysisReport."""

import json
import logging
from pathlib import Path

def run_attention_autotuning():
    """Run attention autotuning across frameworks."""
    from tools.autotune_attention import AttentionAutotuner
    
    logging.basicConfig(level=logging.INFO)
    print("=== Running Attention Autotuning ===")
    
    # Initialize autotuner with realistic parameters
    seq_len = 512
    batch_size = 32
    embed_dim = 512
    
    autotuner = AttentionAutotuner(seq_len, batch_size, embed_dim)
    
    # Run autotuning with fewer iterations for faster demo
    results = autotuner.autotune(num_iterations=20)
    
    # Save results
    results_file = "attention_autotune_results.json"
    autotuner.save_results(results_file)
    
    # Print best configurations
    print("\n=== Best Configurations ===")
    for framework in results:
        config, result = autotuner.get_best_config(framework)
        if result:
            print(f"\n{framework.upper()}:")
            print(f"  Throughput: {result.throughput:.2f} sequences/s")
            print(f"  Forward time: {result.forward_time*1000:.2f} ms")
            print(f"  Memory: {result.memory_usage:.2f} MB")
            print(f"  Config: head_dim={config.head_dim}, num_heads={config.num_heads}")
    
    return results_file

def generate_kernel_analysis_report(results_file: str):
    """Generate comprehensive kernel analysis report."""
    from analysis import KernelAnalysisReport
    
    print("\n=== Generating Kernel Analysis Report ===")
    
    # Load benchmark results
    try:
        with open(results_file, "r") as f:
            benchmark_results = json.load(f)
    except FileNotFoundError:
        print(f"Results file {results_file} not found. Creating example data.")
        # Create example data for demonstration
        benchmark_results = {
            "pytorch": {
                "best_result": {
                    "config": {
                        "head_dim": 64,
                        "num_heads": 8,
                        "dropout_rate": 0.1,
                        "use_flash_attention": True
                    },
                    "framework": "pytorch",
                    "forward_time": 0.005,
                    "memory_usage": 128.5,
                    "throughput": 6400,
                    "flops": 8.5e9
                }
            },
            "jax": {
                "best_result": {
                    "config": {
                        "head_dim": 64,
                        "num_heads": 8,
                        "dropout_rate": 0.0,
                        "precision": "fp16"
                    },
                    "framework": "jax",
                    "forward_time": 0.0045,
                    "memory_usage": 112.3,
                    "throughput": 7111,
                    "flops": 9.2e9
                }
            }
        }
    
    # Create report generator
    report_gen = KernelAnalysisReport("analysis_reports")
    
    # Generate comprehensive reports
    reports = report_gen.generate_reports(benchmark_results)
    
    print("\n=== Analysis Reports Generated ===")
    for report_type, path in reports.items():
        print(f"{report_type.upper()}: {path}")
    
    return reports

def main():
    """Main function to run complete analysis pipeline."""
    print("Kernel Analysis Pipeline Demo")
    print("=============================")
    
    # Step 1: Run attention autotuning
    try:
        results_file = run_attention_autotuning()
    except ImportError as e:
        print(f"Autotuning skipped due to missing dependencies: {e}")
        results_file = "attention_autotune_results.json"  # Will create example data
    
    # Step 2: Generate kernel analysis report
    try:
        reports = generate_kernel_analysis_report(results_file)
        
        print("\n=== Analysis Complete ===")
        print("Check the following files:")
        for report_type, path in reports.items():
            if Path(path).exists():
                print(f"- {path} ({Path(path).stat().st_size} bytes)")
        
        print("\nGenerated plots in analysis_reports/plots/:")
        plots_dir = Path("analysis_reports/plots")
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                print(f"- {plot_file.name}")
    
    except Exception as e:
        print(f"Error generating reports: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
