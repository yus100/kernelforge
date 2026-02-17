"""
Cross-framework Attention Autotuning Module

Supports PyTorch, JAX, and TensorFlow attention mechanisms with hyperparameter optimization.
"""

import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism hyperparameters."""
    head_dim: int = 64
    num_heads: int = 8
    dropout_rate: float = 0.1
    use_flash_attention: bool = True
    block_size: int = 128
    causal: bool = False
    scale_factor: Optional[float] = None
    kernel_type: str = "standard"  # standard, flash, memory_efficient
    precision: str = "fp16"  # fp16, fp32, bf16


@dataclass
class BenchmarkResult:
    """Results from attention mechanism benchmarking."""
    config: AttentionConfig
    framework: str
    forward_time: float
    backward_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    flops: float = 0.0
    power_consumption: float = 0.0
    occupancy: float = 0.0


class AttentionBenchmark(ABC):
    """Abstract base class for attention benchmarking across frameworks."""
    
    def __init__(self, seq_len: int, batch_size: int, embed_dim: int):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.embed_dim = embed_dim
    
    @abstractmethod
    def setup_attention(self, config: AttentionConfig):
        """Setup attention mechanism with given configuration."""
        pass
    
    @abstractmethod
    def benchmark_forward(self, num_iterations: int = 100) -> float:
        """Benchmark forward pass."""
        pass
    
    @abstractmethod
    def benchmark_backward(self, num_iterations: int = 100) -> float:
        """Benchmark backward pass."""
        pass
    
    @abstractmethod
    def measure_memory(self) -> float:
        """Measure memory usage."""
        pass


class PyTorchAttentionBenchmark(AttentionBenchmark):
    """PyTorch attention benchmarking implementation."""
    
    def __init__(self, seq_len: int, batch_size: int, embed_dim: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__(seq_len, batch_size, embed_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention = None
        self.query = None
        self.key = None
        self.value = None
    
    def setup_attention(self, config: AttentionConfig):
        """Setup PyTorch attention mechanism."""
        self.config = config
        
        # Create input tensors
        dtype = torch.float16 if config.precision == "fp16" else torch.float32
        self.query = torch.randn(
            self.batch_size, config.num_heads, self.seq_len, config.head_dim,
            device=self.device, dtype=dtype, requires_grad=True
        )
        self.key = torch.randn_like(self.query, requires_grad=True)
        self.value = torch.randn_like(self.query, requires_grad=True)
        
        # Setup attention mechanism
        if config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            self.attention_func = self._flash_attention
        else:
            self.attention_func = self._standard_attention
    
    def _standard_attention(self):
        """Standard attention implementation."""
        scale = self.config.scale_factor or (self.config.head_dim ** -0.5)
        scores = torch.matmul(self.query, self.key.transpose(-2, -1)) * scale
        
        if self.config.causal:
            mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        if self.config.dropout_rate > 0:
            attn_weights = F.dropout(attn_weights, p=self.config.dropout_rate)
        
        output = torch.matmul(attn_weights, self.value)
        return output
    
    def _flash_attention(self):
        """Flash attention implementation using PyTorch's optimized function."""
        return F.scaled_dot_product_attention(
            self.query, self.key, self.value,
            is_causal=self.config.causal,
            dropout_p=self.config.dropout_rate if self.training else 0.0
        )
    
    def benchmark_forward(self, num_iterations: int = 100) -> float:
        """Benchmark forward pass."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                output = self.attention_func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        return total_time / num_iterations
    
    def benchmark_backward(self, num_iterations: int = 100) -> float:
        """Benchmark backward pass."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_iterations):
            output = self.attention_func()
            loss = output.sum()
            loss.backward()
            # Clear gradients
            for tensor in [self.query, self.key, self.value]:
                if tensor.grad is not None:
                    tensor.grad.zero_()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        return total_time / num_iterations
    
    def measure_memory(self) -> float:
        """Measure memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        return 0.0


class JAXAttentionBenchmark(AttentionBenchmark):
    """JAX attention benchmarking implementation."""
    
    def __init__(self, seq_len: int, batch_size: int, embed_dim: int):
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available")
        super().__init__(seq_len, batch_size, embed_dim)
        self.attention_func = None
        self.query = None
        self.key = None
        self.value = None
    
    def setup_attention(self, config: AttentionConfig):
        """Setup JAX attention mechanism."""
        self.config = config
        
        # Create input arrays
        key = jax.random.PRNGKey(42)
        dtype = jnp.float16 if config.precision == "fp16" else jnp.float32
        
        shape = (self.batch_size, config.num_heads, self.seq_len, config.head_dim)
        self.query = jax.random.normal(key, shape, dtype=dtype)
        self.key = jax.random.normal(key, shape, dtype=dtype)
        self.value = jax.random.normal(key, shape, dtype=dtype)
        
        # Setup and compile attention function
        self.attention_func = jit(self._attention_impl)
    
    def _attention_impl(self, q, k, v):
        """JAX attention implementation."""
        scale = self.config.scale_factor or (self.config.head_dim ** -0.5)
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale
        
        if self.config.causal:
            mask = jnp.triu(jnp.ones_like(scores), k=1)
            scores = scores - mask * 1e9
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attn_weights, v)
        return output
    
    def benchmark_forward(self, num_iterations: int = 100) -> float:
        """Benchmark forward pass."""
        # Warmup
        _ = self.attention_func(self.query, self.key, self.value).block_until_ready()
        
        start_time = time.time()
        for _ in range(num_iterations):
            output = self.attention_func(self.query, self.key, self.value)
            _ = output.block_until_ready()
        
        total_time = time.time() - start_time
        return total_time / num_iterations
    
    def benchmark_backward(self, num_iterations: int = 100) -> float:
        """Benchmark backward pass using JAX grad."""
        def loss_fn(q, k, v):
            output = self.attention_func(q, k, v)
            return jnp.sum(output)
        
        grad_fn = jit(jax.grad(loss_fn, argnums=(0, 1, 2)))
        
        # Warmup
        _ = grad_fn(self.query, self.key, self.value)
        
        start_time = time.time()
        for _ in range(num_iterations):
            grads = grad_fn(self.query, self.key, self.value)
            # Ensure computation completes
            for grad in grads:
                _ = grad.block_until_ready()
        
        total_time = time.time() - start_time
        return total_time / num_iterations
    
    def measure_memory(self) -> float:
        """Measure memory usage (approximation)."""
        total_size = 0
        for arr in [self.query, self.key, self.value]:
            total_size += arr.nbytes
        return total_size / (1024 ** 2)  # MB


class TensorFlowAttentionBenchmark(AttentionBenchmark):
    """TensorFlow attention benchmarking implementation."""
    
    def __init__(self, seq_len: int, batch_size: int, embed_dim: int):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        super().__init__(seq_len, batch_size, embed_dim)
        self.attention_layer = None
        self.query = None
        self.key = None
        self.value = None
    
    def setup_attention(self, config: AttentionConfig):
        """Setup TensorFlow attention mechanism."""
        self.config = config
        
        # Create input tensors
        dtype = tf.float16 if config.precision == "fp16" else tf.float32
        shape = [self.batch_size, self.seq_len, self.embed_dim]
        
        self.query = tf.random.normal(shape, dtype=dtype)
        self.key = tf.random.normal(shape, dtype=dtype)
        self.value = tf.random.normal(shape, dtype=dtype)
        
        # Setup attention layer
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.head_dim,
            dropout=config.dropout_rate,
            use_causal_mask=config.causal
        )
    
    def benchmark_forward(self, num_iterations: int = 100) -> float:
        """Benchmark forward pass."""
        # Warmup
        _ = self.attention_layer(self.query, self.key, self.value)
        
        start_time = time.time()
        for _ in range(num_iterations):
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                output = self.attention_layer(self.query, self.key, self.value)
        
        total_time = time.time() - start_time
        return total_time / num_iterations
    
    def benchmark_backward(self, num_iterations: int = 100) -> float:
        """Benchmark backward pass."""
        start_time = time.time()
        for _ in range(num_iterations):
            with tf.GradientTape() as tape:
                tape.watch([self.query, self.key, self.value])
                output = self.attention_layer(self.query, self.key, self.value)
                loss = tf.reduce_sum(output)
            
            grads = tape.gradient(loss, [self.query, self.key, self.value])
        
        total_time = time.time() - start_time
        return total_time / num_iterations
    
    def measure_memory(self) -> float:
        """Measure memory usage (approximation)."""
        total_size = 0
        for tensor in [self.query, self.key, self.value]:
            total_size += tensor.numpy().nbytes
        return total_size / (1024 ** 2)  # MB


class AttentionAutotuner:
    """Main autotuning class for attention mechanisms across frameworks."""
    
    def __init__(self, seq_len: int, batch_size: int, embed_dim: int):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.results = []
        self.best_config = None
        self.logger = logging.getLogger(__name__)
    
    def generate_config_space(self) -> List[AttentionConfig]:
        """Generate hyperparameter configuration space."""
        configs = []
        
        # Hyperparameter grid
        head_dims = [32, 64, 128]
        num_heads_options = [4, 8, 16]
        dropout_rates = [0.0, 0.1, 0.2]
        kernel_types = ["standard"]
        precisions = ["fp16", "fp32"]
        
        if TORCH_AVAILABLE:
            kernel_types.append("flash")
        
        for head_dim in head_dims:
            for num_heads in num_heads_options:
                if self.embed_dim % (head_dim * num_heads) != 0:
                    continue
                
                for dropout in dropout_rates:
                    for kernel_type in kernel_types:
                        for precision in precisions:
                            config = AttentionConfig(
                                head_dim=head_dim,
                                num_heads=num_heads,
                                dropout_rate=dropout,
                                use_flash_attention=(kernel_type == "flash"),
                                kernel_type=kernel_type,
                                precision=precision
                            )
                            configs.append(config)
        
        return configs
    
    def benchmark_config(self, config: AttentionConfig, framework: str,
                        num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark a specific configuration on a framework."""
        try:
            if framework == "pytorch" and TORCH_AVAILABLE:
                benchmark = PyTorchAttentionBenchmark(
                    self.seq_len, self.batch_size, self.embed_dim
                )
            elif framework == "jax" and JAX_AVAILABLE:
                benchmark = JAXAttentionBenchmark(
                    self.seq_len, self.batch_size, self.embed_dim
                )
            elif framework == "tensorflow" and TF_AVAILABLE:
                benchmark = TensorFlowAttentionBenchmark(
                    self.seq_len, self.batch_size, self.embed_dim
                )
            else:
                raise ValueError(f"Framework {framework} not available")
            
            benchmark.setup_attention(config)
            
            forward_time = benchmark.benchmark_forward(num_iterations)
            backward_time = benchmark.benchmark_backward(num_iterations // 2)
            memory_usage = benchmark.measure_memory()
            
            # Calculate throughput (sequences per second)
            throughput = self.batch_size / forward_time
            
            # Estimate FLOPS (simplified calculation)
            flops = self._estimate_flops(config) / forward_time
            
            result = BenchmarkResult(
                config=config,
                framework=framework,
                forward_time=forward_time,
                backward_time=backward_time,
                memory_usage=memory_usage,
                throughput=throughput,
                flops=flops
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error benchmarking config {config} on {framework}: {e}")
            return None
    
    def _estimate_flops(self, config: AttentionConfig) -> float:
        """Estimate FLOPS for attention computation."""
        # Simplified FLOP estimation for attention
        # Q @ K^T: batch_size * num_heads * seq_len * seq_len * head_dim
        # Softmax: batch_size * num_heads * seq_len * seq_len
        # Attn @ V: batch_size * num_heads * seq_len * seq_len * head_dim
        
        qk_flops = (self.batch_size * config.num_heads * 
                   self.seq_len * self.seq_len * config.head_dim)
        softmax_flops = (self.batch_size * config.num_heads * 
                        self.seq_len * self.seq_len)
        av_flops = qk_flops  # Same as QK
        
        return 2 * (qk_flops + av_flops) + softmax_flops  # Factor of 2 for multiply-add
    
    def autotune(self, frameworks: List[str] = None, 
                 num_iterations: int = 100) -> Dict[str, BenchmarkResult]:
        """Run autotuning across frameworks and configurations."""
        if frameworks is None:
            frameworks = []
            if TORCH_AVAILABLE:
                frameworks.append("pytorch")
            if JAX_AVAILABLE:
                frameworks.append("jax")
            if TF_AVAILABLE:
                frameworks.append("tensorflow")
        
        configs = self.generate_config_space()
        results = {}
        
        self.logger.info(f"Starting autotuning with {len(configs)} configurations "
                        f"across {len(frameworks)} frameworks")
        
        for framework in frameworks:
            framework_results = []
            best_result = None
            
            for i, config in enumerate(configs):
                self.logger.info(f"Benchmarking {framework} config {i+1}/{len(configs)}")
                
                result = self.benchmark_config(config, framework, num_iterations)
                if result:
                    framework_results.append(result)
                    
                    # Track best configuration (optimize for throughput)
                    if best_result is None or result.throughput > best_result.throughput:
                        best_result = result
            
            results[framework] = {
                'all_results': framework_results,
                'best_result': best_result
            }
            
            if best_result:
                self.logger.info(f"Best {framework} config: "
                               f"throughput={best_result.throughput:.2f} seq/s, "
                               f"memory={best_result.memory_usage:.2f} MB")
        
        self.results = results
        return results
    
    def save_results(self, filepath: str):
        """Save autotuning results to JSON file."""
        serializable_results = {}
        
        for framework, data in self.results.items():
            serializable_results[framework] = {
                'all_results': [asdict(r) for r in data['all_results']],
                'best_result': asdict(data['best_result']) if data['best_result'] else None
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def get_best_config(self, framework: str = None) -> Tuple[AttentionConfig, BenchmarkResult]:
        """Get the best configuration overall or for a specific framework."""
        if framework:
            if framework in self.results and self.results[framework]['best_result']:
                best_result = self.results[framework]['best_result']
                return best_result.config, best_result
            else:
                return None, None
        
        # Find best across all frameworks
        best_result = None
        for fw_results in self.results.values():
            if fw_results['best_result']:
                if best_result is None or fw_results['best_result'].throughput > best_result.throughput:
                    best_result = fw_results['best_result']
        
        if best_result:
            return best_result.config, best_result
        return None, None


def main():
    """Example usage of the attention autotuner."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize autotuner
    seq_len = 512
    batch_size = 32
    embed_dim = 512
    
    autotuner = AttentionAutotuner(seq_len, batch_size, embed_dim)
    
    # Run autotuning
    results = autotuner.autotune(num_iterations=50)
    
    # Save results
    autotuner.save_results("attention_autotune_results.json")
    
    # Print best configurations
    print("\n=== Best Configurations ===")
    for framework in results:
        config, result = autotuner.get_best_config(framework)
        if result:
            print(f"\n{framework.upper()}:")
            print(f"  Throughput: {result.throughput:.2f} sequences/s")
            print(f"  Forward time: {result.forward_time*1000:.2f} ms")
            print(f"  Memory: {result.memory_usage:.2f} MB")
            print(f"  Config: {config}")


if __name__ == "__main__":
    main()