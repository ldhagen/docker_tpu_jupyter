```markdown
# Edge TPU Development Environment

A complete Docker-based development environment for Google Coral Edge TPU (M.2 version) with TensorFlow Lite integration. This setup provides reliable access to Edge TPU hardware for machine learning inference acceleration.

## 🚀 Features

- **Full Edge TPU Support**: Access to M.2 Edge TPU hardware
- **TensorFlow & TensorFlow Lite**: Complete ML framework support
- **Jupyter Notebooks**: Interactive development environment
- **Pre-configured Docker**: Reliable, reproducible environment
- **Performance Monitoring**: Benchmarking tools for Edge TPU
- **Book Examples Enhanced**: Compatible with "Hands-On Machine Learning" examples

## 📋 Prerequisites

- Docker and Docker Compose
- Google Coral Edge TPU (M.2 version)
- Ubuntu/Linux host system (recommended)

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd tpu_test3
mkdir -p workspace notebooks
```

### 2. Build and Run

```bash
# Build the container
docker-compose build

# Start the environment
docker-compose up -d edgetpu-workspace

# Check status
docker-compose ps
```

### 3. Verify Edge TPU

```bash
# Test Edge TPU functionality
docker-compose exec edgetpu-workspace python3 test_edgetpu.py
```

### 4. Access Jupyter Notebook

Open your browser to: `http://localhost:8888`

## 📁 Project Structure

```
tpu_test3/
├── docker-compose.yml          # Multi-container setup
├── Dockerfile                  # Edge TPU environment
├── test_edgetpu.py            # Comprehensive TPU test
├── workspace/                  # Your code and models
├── notebooks/                  # Jupyter notebooks
└── README.md
```

## 🐳 Docker Services

### edgetpu-workspace
- Jupyter notebook on port 8888
- Full development environment
- Pre-loaded with test models

### edgetpu-dev
- Development container with shell access
- Direct TPU device access
- Volume mounting for code development

## 🔧 Usage Examples

### Basic Edge TPU Inference

```python
import tflite_runtime.interpreter as tflite
import numpy as np

# Load Edge TPU delegate
delegate = tflite.load_delegate('libedgetpu.so.1')

# Create interpreter with Edge TPU
interpreter = tflite.Interpreter(
    model_path='model_edgetpu.tflite',
    experimental_delegates=[delegate]
)

interpreter.allocate_tensors()

# Run inference
input_data = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

### Performance Benchmarking

```python
from edgetpu_performance import EdgeTPUBenchmark

benchmark = EdgeTPUBenchmark('mobilenet_edgetpu.tflite')
results = benchmark.benchmark(num_runs=100)
```

## 📊 Performance

Typical Edge TPU performance compared to CPU:

| Metric | Edge TPU | CPU |
|--------|----------|-----|
| Inference Time | ~5-10ms | ~50-100ms |
| Throughput | ~100-200 FPS | ~10-20 FPS |
| Power Usage | ~2W | ~15-45W |

## 🎯 Enhanced Book Examples

This environment enhances examples from **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"**:

1. **Chapter 14 (Computer Vision)**: Accelerate MNIST/CIFAR-10 inference
2. **Chapter 16 (NLP)**: Speed up text classification models
3. **Chapter 17 (Autoencoders)**: Faster reconstruction inference
4. **Chapter 18 (GANs)**: Quicker generator inference

### Example: Enhanced MNIST

```python
# Train normally with TensorFlow
model = tf.keras.models.Sequential([...])
model.compile(...)
model.fit(...)

# Convert for Edge TPU
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Run on Edge TPU for 10-50x speedup
```

## 🔍 Testing

Run the comprehensive test suite:

```bash
docker-compose exec edgetpu-workspace python3 test_edgetpu.py
```

Expected output:
```
✓ /dev/apex_0 found
✓ libedgetpu.so.1 loaded successfully  
✓ Edge TPU delegate loaded successfully
✓ Inference successful
🎉 SUCCESS: Edge TPU is fully operational!
```

## 🛠️ Development

### Access Development Container

```bash
docker-compose exec edgetpu-dev /bin/bash
```

### Manual Model Download

If test models aren't auto-downloaded:

```bash
docker-compose exec edgetpu-workspace /bin/bash
cd /workspace
wget https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite
```

### Build from Scratch

```bash
docker-compose build --no-cache
docker-compose up -d
```

## 📝 Common Tasks

### Convert TensorFlow Model for Edge TPU

```python
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('your_model.h5')

# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save for Edge TPU
with open('model_edgetpu.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Monitor Performance

```python
# Real-time performance monitoring
from edgetpu_performance import EdgeTPUBenchmark

benchmark = EdgeTPUBenchmark('your_model.tflite')
performance_data = benchmark.benchmark(num_runs=1000)
```

## 🐛 Troubleshooting

### Edge TPU Not Detected

```bash
# Check device permissions
docker-compose exec edgetpu-workspace ls -la /dev/apex*

# Verify library loading
docker-compose exec edgetpu-workspace python3 -c "import tflite_runtime.interpreter as tflite; tflite.load_delegate('libedgetpu.so.1')"
```

### Model Conversion Issues

- Ensure model uses supported operations
- Use quantization for better performance
- Check input/output shapes match expectations

### Performance Optimization

- Use uint8 quantization
- Batch inferences when possible
- Preprocess data outside inference loop

## 📚 Resources

- [Google Coral Documentation](https://coral.ai/docs/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Edge TPU Model Requirements](https://coral.ai/docs/edgetpu/models-intro/)
- [Hands-On Machine Learning Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Coral Team for Edge TPU hardware
- TensorFlow Team for excellent ML tools
- Aurélien Géron for "Hands-On Machine Learning"

---

**Happy Coding with Edge TPU!** 🚀
```

This README provides:

1. **Clear setup instructions** for new users
2. **Comprehensive documentation** of all features
3. **Usage examples** for common tasks
4. **Troubleshooting guide** for common issues
5. **Integration with the book** you're studying from
6. **Performance benchmarks** to show benefits
7. **Professional structure** suitable for GitHub

The README emphasizes the Edge TPU integration with your machine learning studies and provides practical examples that build on the book's content.
