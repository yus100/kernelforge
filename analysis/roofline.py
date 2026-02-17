import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from .metrics import RooflineData, KernelMetrics

class RooflineAnalyzer:
    def __init__(self, gpu_specs: Dict[str, Any]):
        self.gpu_specs = gpu_specs
        self.peak_compute_fp32 = self._estimate_peak_compute_fp32()
        self.peak_compute_fp16 = self._estimate_peak_compute_fp16()
        self.peak_memory_bandwidth = gpu_specs.get("memory_bandwidth", 900)
    
    def _estimate_peak_compute_fp32(self) -> float:
        sm_count = self.gpu_specs.get("sm_count", 80)
        cores_per_sm = self.gpu_specs.get("cores_per_sm", 128)
        base_clock = self.gpu_specs.get("base_clock", 1500) / 1000.0
        return sm_count * cores_per_sm * base_clock * 2 / 1000.0
    
    def _estimate_peak_compute_fp16(self) -> float:
        return self._estimate_peak_compute_fp32() * 2
    
    def analyze_kernel(self, kernel_metrics: KernelMetrics) -> RooflineData:
        if kernel_metrics.memory_accesses > 0:
            bytes_accessed = kernel_metrics.memory_accesses * 4
            arithmetic_intensity = kernel_metrics.flops / bytes_accessed
        else:
            arithmetic_intensity = 1.0
        
        if kernel_metrics.execution_time > 0:
            achieved_performance = kernel_metrics.flops / kernel_metrics.execution_time / 1e9
        else:
            achieved_performance = 0.0
        
        return RooflineData(
            peak_compute_fp32=self.peak_compute_fp32,
            peak_compute_fp16=self.peak_compute_fp16,
            peak_memory_bandwidth=self.peak_memory_bandwidth,
            arithmetic_intensity=[arithmetic_intensity],
            achieved_performance=[achieved_performance],
            kernel_names=[kernel_metrics.kernel_name]
        )
    
    def plot_roofline(self, roofline_data: RooflineData, save_path: str):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ai_range = np.logspace(-2, 2, 100)
        memory_bound = ai_range * roofline_data.peak_memory_bandwidth
        compute_bound_fp32 = np.full_like(ai_range, roofline_data.peak_compute_fp32)
        compute_bound_fp16 = np.full_like(ai_range, roofline_data.peak_compute_fp16)
        
        ax.loglog(ai_range, memory_bound, "b--", label="Memory Bound", linewidth=2)
        ax.loglog(ai_range, compute_bound_fp32, "r--", label="FP32 Compute Bound", linewidth=2)
        ax.loglog(ai_range, compute_bound_fp16, "g--", label="FP16 Compute Bound", linewidth=2)
        
        for i, (ai, perf, name) in enumerate(zip(
            roofline_data.arithmetic_intensity,
            roofline_data.achieved_performance,
            roofline_data.kernel_names
        )):
            ax.loglog(ai, perf, "ro", markersize=8, label=f"Kernel: {name}")
            ax.annotate(name, (ai, perf), xytext=(5, 5), textcoords="offset points")
        
        ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)")
        ax.set_ylabel("Performance (GFLOPS)")
        ax.set_title("Roofline Model Analysis")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
