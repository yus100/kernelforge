from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RooflineData:
    peak_compute_fp32: float
    peak_compute_fp16: float
    peak_memory_bandwidth: float
    arithmetic_intensity: List[float]
    achieved_performance: List[float]
    kernel_names: List[str]

@dataclass
class MemoryHierarchyMetrics:
    l1_hit_rate: float
    l2_hit_rate: float
    global_memory_throughput: float
    shared_memory_throughput: float
    cache_efficiency: float
    memory_bandwidth_utilization: float
    coalescing_efficiency: float

@dataclass
class OccupancyMetrics:
    theoretical_occupancy: float
    achieved_occupancy: float
    active_warps: int
    active_blocks: int
    registers_per_thread: int
    shared_memory_per_block: int
    occupancy_limited_by: str

@dataclass
class PowerMetrics:
    average_power: float
    peak_power: float
    energy_consumption: float
    temperature: float
    power_efficiency: float
    thermal_throttling: bool

@dataclass
class KernelMetrics:
    kernel_name: str
    execution_time: float
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    roofline: RooflineData
    memory: MemoryHierarchyMetrics
    occupancy: OccupancyMetrics
    power: PowerMetrics
    flops: float
    memory_accesses: int
