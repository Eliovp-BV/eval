## Model Serving

### Starting the Server

To serve the AMD/Llama-3.1-8B-Instruct-FP8-KV model, use the following command:

```bash
python3 /app/wrap_vllm_entrypoint.py \
    --model /app/Llama-3.1-8B-Instruct-FP8-KV/ \
    --served-model-name amd/Llama-3.1-8B-Instruct-FP8-KV \
    --num-scheduler-steps 10 \
    --compilation-config '{"use_cudagraph": false, "cudagraph_capture_sizes": []}' \
    --max-model-len 4096 \
    --kv-cache-dtype fp8
```

To serve the EliovpAI/Deepseek-R1-0528-Qwen3-8B-FP8-KV model, use the following command:

```bash
python3 /app/wrap_vllm_entrypoint.py \
    --model /app/Deepseek-R1-0528-Qwen3-8B-FP8-KV/ \
    --served-model-name EliovpAI/Deepseek-R1-0528-Qwen3-8B-FP8-KV \
    --num-scheduler-steps 10 \
    --compilation-config '{"use_cudagraph": false, "cudagraph_capture_sizes": []}' \
    --kv-cache-dtype fp8
```

### Parameters Explained

- `--model`: Path to the model files
- `--served-model-name`: Name identifier for the served model
- `--num-scheduler-steps`: Number of scheduler steps for request processing
- `--compilation-config`: JSON configuration for CUDA graph optimization
  - `use_cudagraph`: Disabled for compatibility with Paiton to use VLLMs faster Async Engine
  - `cudagraph_capture_sizes`: Empty array for no capture sizes, again for compatibility with Paiton
- `--max-model-len`: Maximum sequence length (4096 tokens)
- `--kv-cache-dtype fp8`: Use FP8 precision for key-value cache (memory optimization)

### Server Features

- **FP8 Key-Value Cache**: Reduces memory usage while maintaining accuracy
- **Paiton Integration**: AMD-optimized inference framework
- **vLLM Backend**: High-performance serving with efficient request handling
- **Custom Entry Point**: Integrates Paiton models with vLLM serving

## Benchmarking

### Running Performance Benchmarks

To benchmark the model performance using the ShareGPT dataset:

```bash
python3 /app/vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model amd/Llama-3.1-8B-Instruct-FP8-KV \
    --dataset-name sharegpt \
    --dataset-path /app/vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1024 \
    --random-range-ratio 1.0 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --sharegpt-output-len 256
```

### Benchmark Parameters Explained

- `--backend vllm`: Use vLLM as the serving backend
- `--model`: Model identifier for benchmarking
- `--dataset-name sharegpt`: Use ShareGPT dataset for testing
- `--dataset-path`: Path to the ShareGPT dataset file
- `--num-prompts 1024`: Number of prompts to test
- `--random-range-ratio 1.0`: Use full range of dataset
- `--percentile-metrics`: Metrics to measure:
  - `ttft`: Time to first token
  - `tpot`: Time per output token
  - `itl`: Inter-token latency
  - `e2el`: End-to-end latency
- `--sharegpt-output-len 256`: Generate 256 tokens per response

### Expected Metrics

The benchmark will provide detailed performance metrics including:
- **Throughput**: Requests per second
- **Latency**: Response time measurements
- **Memory Usage**: GPU memory consumption
- **Token Generation Speed**: Tokens per second

## Model Architecture

### Key Features

1. **FP8 Key-Value Cache**: 
   - Reduces memory footprint by ~50% compared to FP16
   - Maintains model quality with minimal accuracy loss
   - Optimized for AMD GPUs

2. **Paiton Integration**:
   - AMD-optimized inference kernels
   - Efficient memory management
   - Hardware-specific optimizations

3. **vLLM Serving**:
   - High-throughput request handling
   - Efficient batching and scheduling
   - RESTful API interface

## API Usage

Once the server is running, you can interact with it using the OpenAI-compatible API:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="amd/Llama-3.1-8B-Instruct-FP8-KV",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Performance Optimization

### Memory Optimization
- FP8 KV cache reduces memory usage significantly
- Efficient memory allocation through Paiton
- Dynamic memory management in vLLM

### Throughput Optimization
- Request batching for higher throughput
- Efficient scheduler with configurable steps
- Hardware-specific optimizations for AMD GPUs

### Latency Optimization
- Optimized token generation pipeline
- Efficient attention computation
- Reduced memory bandwidth requirements

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce `--max-model-len`
   - Ensure sufficient GPU memory
   - Check FP8 compatibility

2. **Model Loading Issues**:
   - Verify model path is correct
   - Check model file integrity
   - Ensure Paiton compatibility

3. **Performance Issues**:
   - Adjust `--num-scheduler-steps`
   - Monitor GPU utilization
   - Check compilation configuration

## Monitoring

### Key Metrics to Monitor

- **GPU Memory Usage**: Monitor with `rocm-smi`
- **Throughput**: Requests per second
- **Latency**: P50, P95, P99 latencies
- **Token Generation Speed**: Tokens per second

### Health Checks

The server provides health endpoints:
- `GET /health`: Basic health check
- `GET /v1/models`: List available models
- `GET /metrics`: Performance metrics (if enabled)
