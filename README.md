```markdown
# Edge TPU Development Environment

# 🚨 CRITICAL KERNEL REQUIREMENT 🚨

## **YOUR HOST SYSTEM MUST BOOT WITH LINUX KERNEL 6.8+ FOR EDGE TPU DRIVERS TO WORK**

### **This is a HARD REQUIREMENT - Later kernels will NOT work with Edge TPU M.2**

**Check your kernel version immediately:**
```bash
uname -r
```


A complete Docker-based development environment for Google Coral Edge TPU (M.2 version) with TensorFlow Lite integration. This setup provides reliable access to Edge TPU hardware for machine learning inference acceleration.

## ⚠️ IMPORTANT: Critical Installation Notes

**Before proceeding, read this section carefully to avoid common installation issues!**

### The Kernel Version Requirement Issue

**Most Common Failure Cause:** Using Linux kernel > 6.8 with Edge TPU M.2

**Symptoms:**
- `/dev/apex_0` device never appears
- Host drivers install but device not detected
- `ls /dev/apex*` returns nothing
- Docker container cannot find Edge TPU device

### The PyCoral Package Conflict Issue

There are **two different Python packages** named `pycoral` that conflict with each other:

1. **Correct package**: `python3-pycoral` from Google (system package)
2. **Wrong package**: `pycoral` from PyPI (pip package) - this is a different library with the same name!

Using the wrong package will cause `ModuleNotFoundError: No module named 'pycoral.utils'` errors.

## 🚀 Features

- **Full Edge TPU Support**: Access to M.2 Edge TPU hardware via `/dev/apex_0` (KERNEL 6.8+ REQUIRED)
- **TensorFlow & TensorFlow Lite**: Complete ML framework support
- **Jupyter Notebooks**: Interactive development environment
- **Pre-configured Docker**: Reliable, reproducible environment
- **Performance Monitoring**: Benchmarking tools for Edge TPU
- **Book Examples Enhanced**: Compatible with "Hands-On Machine Learning" examples

## 📋 Prerequisites

- **LINUX KERNEL 6.8 (MANDATORY FOR EDGE TPU M.2)
- Docker and Docker Compose
- Google Coral Edge TPU (M.2 version)
- Ubuntu/Linux host system (recommended)
- **Host must have Edge TPU drivers installed**

## 🛠️ Quick Start

### 0. VERIFY KERNEL VERSION FIRST!

```bash
# ⚠️ STOP! Check this first!
uname -r

# Must output something starting with 6.8.x, 
# If not, STOP and downgrade your kernel before continuing!
```

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd docker_tpu_jupyter
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

## 🔧 CRITICAL: Installation Troubleshooting

### Issue 0: Kernel Version Too New (MOST COMMON ISSUE)

**Symptoms:**
- `/dev/apex_0` never appears on host system
- `lsusb | grep -i coral` shows nothing
- Host driver installation succeeds but device not detected

**Root Cause:** Linux kernel > 6.8 lacks proper M.2 Edge TPU support.


### Issue 1: PyCoral Module Not Found

**Symptoms:**
```python
>>> from pycoral.utils.edgetpu import list_edge_tpus
ModuleNotFoundError: No module named 'pycoral.utils'
```

**Root Cause:** Wrong `pycoral` package installed from PyPI instead of Google's repository.

**Solution:**
```bash
# Remove the incorrect pip package
pip uninstall pycoral

# Install the correct system package
apt-get remove --purge -y python3-pycoral
apt-get autoremove -y
apt-get install -y python3-pycoral

# Verify correct installation
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print('SUCCESS!')"
```

### Issue 2: Edge TPU Device Not Accessible

**Symptoms:**
```python
FileNotFoundError: [Errno 2] No such file or directory: '/dev/apex_0'
```

**Root Cause:** Docker container doesn't have device access, host drivers missing, or **KERNEL TOO OLD**.

**Solutions:**

**First: Check Kernel Version (Most Likely Cause)**
```bash
# ⚠️ Check this first!
uname -r
# If not 6.8+, see "Issue 0" above
```

**Option A: Check host drivers**
```bash
# On host system, check if Edge TPU is detected
lsusb | grep -i coral
lspci | grep -i coral
ls /dev/apex*

# Install host drivers if missing
# For Ubuntu:
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

**Option B: Fix Docker device mapping**
```bash
# Ensure your docker-compose.yml has:
# devices:
#   - "/dev/apex_0:/dev/apex_0"

# Or run manually with:
docker run --device /dev/apex_0:/dev/apex_0 your-image


### The PyCoral Package Conflict Issue

There are **two different Python packages** named `pycoral` that conflict with each other:

1. **Correct package**: `python3-pycoral` from Google (system package)
2. **Wrong package**: `pycoral` from PyPI (pip package) - this is a different library with the same name!

Using the wrong package will cause `ModuleNotFoundError: No module named 'pycoral.utils'` errors.

## 🚀 Features

- **Full Edge TPU Support**: Access to M.2 Edge TPU hardware via `/dev/apex_0`
- **TensorFlow & TensorFlow Lite**: Complete ML framework support
- **Jupyter Notebooks**: Interactive development environment
- **Pre-configured Docker**: Reliable, reproducible environment
- **Performance Monitoring**: Benchmarking tools for Edge TPU
- **Book Examples Enhanced**: Compatible with "Hands-On Machine Learning" examples

## 📋 Prerequisites

- Docker and Docker Compose
- Google Coral Edge TPU (M.2 version)
- Ubuntu/Linux host system (recommended)
- **Host must have Edge TPU drivers installed**

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd docker_tpu_jupyter
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

## 🔧 CRITICAL: Installation Troubleshooting

### Issue 1: PyCoral Module Not Found

**Symptoms:**
```python
>>> from pycoral.utils.edgetpu import list_edge_tpus
ModuleNotFoundError: No module named 'pycoral.utils'
```

**Root Cause:** Wrong `pycoral` package installed from PyPI instead of Google's repository.

**Solution:**
```bash
# Remove the incorrect pip package
pip uninstall pycoral

# Install the correct system package
apt-get remove --purge -y python3-pycoral
apt-get autoremove -y
apt-get install -y python3-pycoral

# Verify correct installation
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print('SUCCESS!')"
```

### Issue 2: Edge TPU Device Not Accessible

**Symptoms:**
```python
FileNotFoundError: [Errno 2] No such file or directory: '/dev/apex_0'
```

**Root Cause:** Docker container doesn't have device access or host drivers missing.

**Solutions:**

**Option A: Check host drivers**
```bash
# On host system, check if Edge TPU is detected
lsusb | grep -i coral
lspci | grep -i coral
ls /dev/apex*

# Install host drivers if missing
# For Ubuntu:
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

**Option B: Fix Docker device mapping**
```bash
# Ensure your docker-compose.yml has:
# devices:
#   - "/dev/apex_0:/dev/apex_0"

# Or run manually with:
docker run --device /dev/apex_0:/dev/apex_0 your-image
```

### Issue 3: Library Loading Failures

**Symptoms:**
```python
RuntimeError: Failed to load delegate from libedgetpu.so.1
```

**Solutions:**
```bash
# Install Edge TPU runtime library
apt-get install -y libedgetpu1-std

# Check library location
ldconfig -p | grep edgetpu

# Alternative: Use direct path
delegate = tflite.load_delegate('/usr/lib/x86_64-linux-gnu/libedgetpu.so.1')
```

### Issue 4: Model Compatibility Issues

**Symptoms:**
```python
RuntimeError: Failed to prepare for TPU compilation
```

**Solutions:**
- Ensure model is quantized (uint8)
- Check model uses supported operations
- Convert model properly for Edge TPU:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
```

## 📁 Project Structure

```
docker_tpu_jupyter/
├── docker-compose.yml            # Multi-container setup
├── Dockerfile                    # Edge TPU environment
├── /workspace/test_edgetpu.py    # Comprehensive TPU test
├── /workspace/verify_edgetpu.py  # TPU verify
├── workspace/                    # Your code and models
├── notebooks/                    # Jupyter notebooks
└── README.md
```

## 🐳 Docker Services

### edgetpu-workspace
- Jupyter notebook on port 8888
- Full development environment
- Pre-loaded with test models
- **Pre-configured with correct pycoral installation**

### edgetpu-dev
- Development container with shell access
- Direct TPU device access
- Volume mounting for code development

## 🛠️ Comprehensive Troubleshooting Script

We provide `fix_pycoral_conflict.py` to automatically diagnose and fix common issues:

```bash
docker-compose exec edgetpu-workspace python3 fix_pycoral_conflict.py
```

This script checks:
- Correct pycoral package installation
- Edge TPU device accessibility
- Library availability
- Model compatibility

## 🔧 Usage Examples

### Basic Edge TPU Inference (Reliable Method)

```python
import tflite_runtime.interpreter as tflite
import numpy as np

def safe_edgetpu_inference(model_path, input_data):
    """Safe Edge TPU inference with error handling"""
    try:
        # Load Edge TPU delegate
        delegate = tflite.load_delegate('libedgetpu.so.1')
        
        # Create interpreter with Edge TPU
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate]
        )
        
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        return output
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return None

# Usage
input_data = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.uint8)
result = safe_edgetpu_inference('model_edgetpu.tflite', input_data)
```

### Alternative: Fallback to CPU

```python
def robust_inference(model_path, input_data, use_edgetpu=True):
    """Robust inference with Edge TPU fallback to CPU"""
    try:
        if use_edgetpu:
            delegate = tflite.load_delegate('libedgetpu.so.1')
            interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[delegate]
            )
        else:
            interpreter = tflite.Interpreter(model_path=model_path)
            
        interpreter.allocate_tensors()
        # ... rest of inference code
        
    except Exception as e:
        print(f"Edge TPU failed, falling back to CPU: {e}")
        return robust_inference(model_path, input_data, use_edgetpu=False)
```

## 📊 Performance

Typical Edge TPU performance compared to CPU:

| Metric | Edge TPU | CPU | Notes |
|--------|----------|-----|-------|
| Inference Time | ~5-10ms | ~50-100ms | Quantized models |
| Throughput | ~100-200 FPS | ~10-20 FPS | Batch size 1 |
| Power Usage | ~2W | ~15-45W | Significant savings |

## 🎯 Enhanced Book Examples

This environment enhances examples from **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** with specific troubleshooting for Edge TPU compatibility.

### Fixed Example: Enhanced MNIST with Edge TPU

```python
import tensorflow as tf
import numpy as np

def create_edgetpu_compatible_model():
    """Create a model compatible with Edge TPU constraints"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def convert_for_edgetpu(model, representative_data):
    """Proper conversion for Edge TPU with error handling"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    
    # Critical: These settings enable Edge TPU compatibility
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    try:
        tflite_model = converter.convert()
        with open('mnist_edgetpu.tflite', 'wb') as f:
            f.write(tflite_model)
        print("✓ Model successfully converted for Edge TPU")
        return True
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False
```

## 🔍 Comprehensive Testing

Run the complete test suite:

```bash
docker-compose exec edgetpu-workspace python3 test_edgetpu.py
```

**Expected output when working:**
```
✓ /dev/apex_0 found and accessible
✓ libedgetpu.so.1 loaded successfully  
✓ Edge TPU delegate loaded successfully
✓ Basic inference test passed
🎉 SUCCESS: Edge TPU is fully operational!
```

**Expected output when troubleshooting needed:**
```
⚠️  Issues detected:
- PyCoral package conflict found
- Running automatic fixes...
✓ Removed incorrect pip package
✓ Installed correct system package
✓ Verified Edge TPU functionality
🎉 All issues resolved!
```

## 🛠️ Development Workflow

### 1. Initial Setup Checklist
- [ ] Verify host Edge TPU drivers
- [ ] Build Docker containers
- [ ] Run diagnostic script
- [ ] Test basic inference

### 2. Daily Development
```bash
# Start environment
docker-compose up -d edgetpu-workspace

# Run tests
docker-compose exec edgetpu-workspace python3 test_edgetpu.py

# Access Jupyter: http://localhost:8888
# Or shell access:
docker-compose exec edgetpu-workspace /bin/bash
```

### 3. Model Development Cycle
```python
# 1. Develop and train model (CPU/GPU)
model = create_and_train_model()

# 2. Convert for Edge TPU
success = convert_for_edgetpu(model, representative_data)

# 3. Test on Edge TPU
if success:
    results = test_edgetpu_inference('model_edgetpu.tflite')
```

## 📝 Common Issues & Solutions

### Package Management Issues

**Problem:** Mixed pip and apt packages causing conflicts
**Solution:** Always use `apt-get install python3-pycoral`, never `pip install pycoral`

### Device Permission Issues

**Problem:** `Permission denied` when accessing `/dev/apex_0`
**Solution:** Ensure Docker has proper device mapping and user is in correct groups

### Model Conversion Failures

**Problem:** Model uses unsupported operations
**Solution:** Use Edge TPU-compatible operations and proper quantization

### Performance Issues

**Problem:** Inference slower than expected
**Solution:** Ensure proper quantization, batch processing, and input preprocessing

## 🚨 Emergency Recovery

If everything breaks:

```bash
# Nuclear option: complete rebuild
docker-compose down
docker system prune -a
docker-compose build --no-cache
docker-compose up -d

# Then run automatic fixes
docker-compose exec edgetpu-workspace python3 fix_pycoral_conflict.py
```

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

**Need Help?** Check the troubleshooting section first, then open an issue with your `test_edgetpu.py` output!

**Happy Coding with Edge TPU!** 🚀
```

