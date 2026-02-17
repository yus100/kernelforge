#!/usr/bin/env python3
# Quick test of the analysis system

import json
from pathlib import Path

def test_kernel_analysis():
    from analysis import KernelAnalysisReport
    
    print("Testing Kernel Analysis Report...")
    
    # Create example benchmark data
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
        }
    }
    
    # Generate reports
    report_gen = KernelAnalysisReport("test_reports")
    reports = report_gen.generate_reports(benchmark_results)
    
    print("Reports generated:")
    for report_type, path in reports.items():
        if Path(path).exists():
            print(f"- {report_type}: {path} ({Path(path).stat().st_size} bytes)")
    
    return True

def main():
    print("Quick Analysis System Test")
    print("==========================")
    
    try:
        test_kernel_analysis()
        print("SUCCESS: Kernel analysis report test passed")
    except Exception as e:
        print(f"ERROR: Kernel analysis test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test complete. Check test_reports/ for generated files.")

if __name__ == "__main__":
    main()
