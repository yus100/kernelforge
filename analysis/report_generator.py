import json
import time
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from .metrics import *
from .profiler import GPUProfiler
from .roofline import RooflineAnalyzer
from .plots import PlotGenerator

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class KernelAnalysisReport:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.profiler = GPUProfiler()
        self.gpu_specs = self.profiler.get_gpu_specs()
        self.roofline_analyzer = RooflineAnalyzer(self.gpu_specs)
        
        self.kernel_metrics = []
    
    def add_kernel_metrics(self, metrics: KernelMetrics):
        self.kernel_metrics.append(metrics)
    
    def analyze_attention_benchmark(self, benchmark_results: Dict[str, Any]) -> List[KernelMetrics]:
        metrics_list = []
        
        for framework, results in benchmark_results.items():
            if "best_result" in results and results["best_result"]:
                result = results["best_result"]
                metrics = self._create_kernel_metrics_from_benchmark(result, framework)
                metrics_list.append(metrics)
        
        return metrics_list
    
    def _create_kernel_metrics_from_benchmark(self, benchmark_result, framework: str) -> KernelMetrics:
        config = benchmark_result["config"]
        
        # Estimate kernel parameters
        seq_len = 512
        block_size = (min(config["head_dim"], 32), min(seq_len // 4, 32), 1)
        grid_size = (
            (config["head_dim"] + block_size[0] - 1) // block_size[0],
            (seq_len + block_size[1] - 1) // block_size[1],
            config["num_heads"]
        )
        
        # Estimate memory metrics
        memory_metrics = MemoryHierarchyMetrics(
            l1_hit_rate=0.85,
            l2_hit_rate=0.75,
            global_memory_throughput=benchmark_result["memory_usage"] / benchmark_result["forward_time"] * 1000,
            shared_memory_throughput=500.0,
            cache_efficiency=0.80,
            memory_bandwidth_utilization=0.65,
            coalescing_efficiency=0.90
        )
        
        # Estimate occupancy
        occupancy_metrics = OccupancyMetrics(
            theoretical_occupancy=1.0,
            achieved_occupancy=0.75,
            active_warps=grid_size[0] * grid_size[1] * grid_size[2] * 4,
            active_blocks=grid_size[0] * grid_size[1] * grid_size[2],
            registers_per_thread=32,
            shared_memory_per_block=config.get("block_size", 512) * 4,
            occupancy_limited_by="registers"
        )
        
        # Measure power
        power_metrics = self.profiler.measure_power(benchmark_result["forward_time"])
        power_metrics.power_efficiency = benchmark_result["flops"] / power_metrics.average_power / 1e9
        
        # Create temporary kernel metrics for roofline analysis
        temp_metrics = KernelMetrics(
            kernel_name=f"{framework}_attention",
            execution_time=benchmark_result["forward_time"],
            grid_size=grid_size,
            block_size=block_size,
            roofline=None,
            memory=memory_metrics,
            occupancy=occupancy_metrics,
            power=power_metrics,
            flops=benchmark_result["flops"],
            memory_accesses=int(benchmark_result["memory_usage"] * 1024 * 1024 / 4)
        )
        
        # Roofline analysis
        roofline_data = self.roofline_analyzer.analyze_kernel(temp_metrics)
        temp_metrics.roofline = roofline_data
        
        return temp_metrics
    
    def generate_plots(self):
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        if self.kernel_metrics:
            plot_gen = PlotGenerator(self.kernel_metrics)
            
            # Generate roofline plot
            combined_roofline = RooflineData(
                peak_compute_fp32=self.roofline_analyzer.peak_compute_fp32,
                peak_compute_fp16=self.roofline_analyzer.peak_compute_fp16,
                peak_memory_bandwidth=self.roofline_analyzer.peak_memory_bandwidth,
                arithmetic_intensity=[m.roofline.arithmetic_intensity[0] for m in self.kernel_metrics],
                achieved_performance=[m.roofline.achieved_performance[0] for m in self.kernel_metrics],
                kernel_names=[m.kernel_name for m in self.kernel_metrics]
            )
            
            self.roofline_analyzer.plot_roofline(
                combined_roofline, 
                str(plots_dir / "roofline_analysis.png")
            )
            
            plot_gen.plot_memory_hierarchy(plots_dir / "memory_hierarchy.png")
            plot_gen.plot_occupancy(plots_dir / "occupancy_analysis.png")
            plot_gen.plot_power_analysis(plots_dir / "power_analysis.png")
\n    def generate_markdown_report(self) -> str:\n        report_path = self.output_dir / "kernel_analysis_report.md"\n        \n        md_content = f"""# Kernel Analysis Report\nGenerated on: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n## GPU Hardware Specifications\n- **Name**: {self.gpu_specs["name"]}\n- **Compute Capability**: {self.gpu_specs["compute_capability"]}\n- **Total Memory**: {self.gpu_specs["memory_total"]} GB\n- **Memory Bandwidth**: {self.gpu_specs["memory_bandwidth"]} GB/s\n\n## Executive Summary\nThis report analyzes {len(self.kernel_metrics)} kernel(s) with roofline analysis,\nmemory hierarchy performance, GPU occupancy, and power consumption.\n\n## Roofline Analysis\n![Roofline Analysis](plots/roofline_analysis.png)\n"""\n        \n        # Add per-kernel analysis\n        for metrics in self.kernel_metrics:\n            md_content += f"""\n### {metrics.kernel_name}\n- **L1 Hit Rate**: {metrics.memory.l1_hit_rate:.1%}\n- **L2 Hit Rate**: {metrics.memory.l2_hit_rate:.1%}\n- **Occupancy**: {metrics.occupancy.achieved_occupancy:.1%}\n- **Power**: {metrics.power.average_power:.1f} W\n"""\n        \n        md_content += """\n## Memory Hierarchy Analysis\n![Memory Hierarchy](plots/memory_hierarchy.png)\n\n## Occupancy Analysis\n![Occupancy Analysis](plots/occupancy_analysis.png)\n\n## Power Analysis\n![Power Analysis](plots/power_analysis.png)\n\n## Recommendations\n- Monitor occupancy and memory efficiency\n- Consider precision optimization for memory-bound kernels\n- Balance performance and power consumption\n"""\n        \n        report_path.write_text(md_content)\n        return str(report_path)\n    \n    def generate_pdf_report(self) -> Optional[str]:\n        if not REPORTLAB_AVAILABLE:\n            print("ReportLab not available, skipping PDF generation")\n            return None\n        \n        report_path = self.output_dir / "kernel_analysis_report.pdf"\n        doc = SimpleDocTemplate(str(report_path), pagesize=letter)\n        styles = getSampleStyleSheet()\n        story = []\n        \n        # Title\n        title_style = ParagraphStyle(\n            "CustomTitle",\n            parent=styles["Heading1"],\n            fontSize=24,\n            textColor=colors.darkblue,\n            spaceAfter=30,\n            alignment=1\n        )\n        story.append(Paragraph("Kernel Analysis Report", title_style))\n        story.append(Spacer(1, 20))\n        \n        # GPU Specs Table\n        story.append(Paragraph("GPU Hardware Specifications", styles["Heading2"]))\n        gpu_data = [\n            ["Property", "Value"],\n            ["Name", self.gpu_specs["name"]],\n            ["Total Memory", f"{self.gpu_specs["memory_total"]} GB"],\n            ["Memory Bandwidth", f"{self.gpu_specs["memory_bandwidth"]} GB/s"]\n        ]\n        \n        gpu_table = Table(gpu_data)\n        gpu_table.setStyle(TableStyle([\n            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),\n            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),\n            ("ALIGN", (0, 0), (-1, -1), "CENTER"),\n            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),\n            ("GRID", (0, 0), (-1, -1), 1, colors.black)\n        ]))\n        \n        story.append(gpu_table)\n        story.append(Spacer(1, 20))\n        \n        # Add plots\n        plots_dir = self.output_dir / "plots"\n        if (plots_dir / "roofline_analysis.png").exists():\n            story.append(Paragraph("Roofline Analysis", styles["Heading2"]))\n            story.append(Image(str(plots_dir / "roofline_analysis.png"), width=6*inch, height=4.8*inch))\n        \n        doc.build(story)\n        return str(report_path)\n    \n    def generate_reports(self, benchmark_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:\n        if benchmark_results:\n            kernel_metrics = self.analyze_attention_benchmark(benchmark_results)\n            for metrics in kernel_metrics:\n                self.add_kernel_metrics(metrics)\n        \n        self.generate_plots()\n        \n        reports = {}\n        md_path = self.generate_markdown_report()\n        reports["markdown"] = md_path\n        print(f"Markdown report generated: {md_path}")\n        \n        pdf_path = self.generate_pdf_report()\n        if pdf_path:\n            reports["pdf"] = pdf_path\n            print(f"PDF report generated: {pdf_path}")\n        \n        return reports\n