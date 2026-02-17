import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from .metrics import KernelMetrics

class PlotGenerator:
    def __init__(self, kernel_metrics: List[KernelMetrics]):
        self.kernel_metrics = kernel_metrics
    
    def plot_memory_hierarchy(self, save_path: Path):
        if not self.kernel_metrics:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        kernels = [m.kernel_name for m in self.kernel_metrics]
        
        # L1/L2 Hit Rates
        l1_rates = [m.memory.l1_hit_rate for m in self.kernel_metrics]
        l2_rates = [m.memory.l2_hit_rate for m in self.kernel_metrics]
        x = np.arange(len(kernels))
        width = 0.35
        
        ax1.bar(x - width/2, l1_rates, width, label="L1 Hit Rate", alpha=0.8)
        ax1.bar(x + width/2, l2_rates, width, label="L2 Hit Rate", alpha=0.8)
        ax1.set_xlabel("Kernel")
        ax1.set_ylabel("Hit Rate")
        ax1.set_title("Cache Hit Rates")
        ax1.set_xticks(x)
        ax1.set_xticklabels(kernels, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Memory Throughput
        global_throughput = [m.memory.global_memory_throughput for m in self.kernel_metrics]
        shared_throughput = [m.memory.shared_memory_throughput for m in self.kernel_metrics]
        
        ax2.bar(x - width/2, global_throughput, width, label="Global Memory", alpha=0.8)
        ax2.bar(x + width/2, shared_throughput, width, label="Shared Memory", alpha=0.8)
        ax2.set_xlabel("Kernel")
        ax2.set_ylabel("Throughput (GB/s)")
        ax2.set_title("Memory Throughput")
        ax2.set_xticks(x)
        ax2.set_xticklabels(kernels, rotation=45, ha="right")
        ax2.legend()
        
        # Memory Efficiency
        bandwidth_util = [m.memory.memory_bandwidth_utilization for m in self.kernel_metrics]
        coalescing_eff = [m.memory.coalescing_efficiency for m in self.kernel_metrics]
        
        ax3.bar(x - width/2, bandwidth_util, width, label="Bandwidth Utilization", alpha=0.8)
        ax3.bar(x + width/2, coalescing_eff, width, label="Coalescing Efficiency", alpha=0.8)
        ax3.set_xlabel("Kernel")
        ax3.set_ylabel("Efficiency")
        ax3.set_title("Memory Efficiency")
        ax3.set_xticks(x)
        ax3.set_xticklabels(kernels, rotation=45, ha="right")
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Overall Cache Efficiency
        cache_eff = [m.memory.cache_efficiency for m in self.kernel_metrics]
        ax4.bar(kernels, cache_eff, alpha=0.8, color="purple")
        ax4.set_xlabel("Kernel")
        ax4.set_ylabel("Cache Efficiency")
        ax4.set_title("Overall Cache Efficiency")
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_occupancy(self, save_path: Path):
        if not self.kernel_metrics:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        kernels = [m.kernel_name for m in self.kernel_metrics]
        
        # Theoretical vs Achieved Occupancy
        theoretical = [m.occupancy.theoretical_occupancy for m in self.kernel_metrics]
        achieved = [m.occupancy.achieved_occupancy for m in self.kernel_metrics]
        x = np.arange(len(kernels))
        width = 0.35
        
        ax1.bar(x - width/2, theoretical, width, label="Theoretical", alpha=0.8)
        ax1.bar(x + width/2, achieved, width, label="Achieved", alpha=0.8)
        ax1.set_xlabel("Kernel")
        ax1.set_ylabel("Occupancy")
        ax1.set_title("Occupancy Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(kernels, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Active Warps
        active_warps = [m.occupancy.active_warps for m in self.kernel_metrics]
        ax2.bar(kernels, active_warps, alpha=0.8, color="orange")
        ax2.set_xlabel("Kernel")
        ax2.set_ylabel("Active Warps")
        ax2.set_title("Active Warps per Kernel")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        # Resource Usage
        registers = [m.occupancy.registers_per_thread for m in self.kernel_metrics]
        ax3.bar(kernels, registers, alpha=0.8, color="green")
        ax3.set_xlabel("Kernel")
        ax3.set_ylabel("Registers per Thread")
        ax3.set_title("Register Usage")
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        
        # Shared Memory Usage
        shared_mem = [m.occupancy.shared_memory_per_block / 1024 for m in self.kernel_metrics]  # KB
        ax4.bar(kernels, shared_mem, alpha=0.8, color="red")
        ax4.set_xlabel("Kernel")
        ax4.set_ylabel("Shared Memory per Block (KB)")
        ax4.set_title("Shared Memory Usage")
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_power_analysis(self, save_path: Path):
        if not self.kernel_metrics:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        kernels = [m.kernel_name for m in self.kernel_metrics]
        
        # Average vs Peak Power
        avg_power = [m.power.average_power for m in self.kernel_metrics]
        peak_power = [m.power.peak_power for m in self.kernel_metrics]
        x = np.arange(len(kernels))
        width = 0.35
        
        ax1.bar(x - width/2, avg_power, width, label="Average", alpha=0.8)
        ax1.bar(x + width/2, peak_power, width, label="Peak", alpha=0.8)
        ax1.set_xlabel("Kernel")
        ax1.set_ylabel("Power (W)")
        ax1.set_title("Power Consumption")
        ax1.set_xticks(x)
        ax1.set_xticklabels(kernels, rotation=45, ha="right")
        ax1.legend()
        
        # Energy Consumption
        energy = [m.power.energy_consumption for m in self.kernel_metrics]
        ax2.bar(kernels, energy, alpha=0.8, color="orange")
        ax2.set_xlabel("Kernel")
        ax2.set_ylabel("Energy (J)")
        ax2.set_title("Energy Consumption")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        # Temperature
        temperatures = [m.power.temperature for m in self.kernel_metrics]
        ax3.bar(kernels, temperatures, alpha=0.8, color="red")
        ax3.set_xlabel("Kernel")
        ax3.set_ylabel("Temperature (°C)")
        ax3.set_title("Operating Temperature")
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        
        # Power Efficiency
        efficiency = [m.power.power_efficiency for m in self.kernel_metrics]
        ax4.bar(kernels, efficiency, alpha=0.8, color="green")
        ax4.set_xlabel("Kernel")
        ax4.set_ylabel("Power Efficiency (GFLOPS/W)")
        ax4.set_title("Power Efficiency")
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
