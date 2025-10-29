#!/usr/bin/env python3
import os
import sys
import urllib.request
import subprocess

def download_model():
    """Download the test model if missing"""
    model_path = 'mobilenet_edgetpu.tflite'
    
    if os.path.exists(model_path):
        print(f"✓ Test model found: {model_path}")
        return True
    
    print("Downloading test model...")
    urls = [
        'https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite',
        'https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite'
    ]
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, model_path)
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                print(f"✓ Model downloaded successfully: {model_path}")
                return True
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Try wget as fallback
    try:
        result = subprocess.run([
            'wget', '-q', 
            'https://github.com/google-coral/test_data/raw/master/mobilenet_v1_1.0_224_quant_edgetpu.tflite',
            '-O', model_path
        ], capture_output=True, text=True)
        
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            print(f"✓ Model downloaded via wget: {model_path}")
            return True
    except:
        pass
    
    print("✗ Could not download test model")
    return False

def test_edgetpu_inference():
    print("\n=== Edge TPU Inference Test ===")
    
    if not download_model():
        return False
    
    try:
        import tflite_runtime.interpreter as tflite
        import numpy as np
        
        # Load with Edge TPU delegate
        delegate = tflite.load_delegate('libedgetpu.so.1')
        interpreter = tflite.Interpreter(
            model_path='mobilenet_edgetpu.tflite',
            experimental_delegates=[delegate]
        )
        
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✓ Model loaded successfully")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        
        # Test inference
        input_shape = input_details[0]['shape']
        input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
        
        print("Running inference...")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✓ Inference successful!")
        print(f"  Output shape: {output_data.shape}")
        print(f"  First 5 values: {output_data[0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False

def benchmark_edgetpu():
    print("\n=== Edge TPU Benchmark ===")
    
    try:
        import tflite_runtime.interpreter as tflite
        import numpy as np
        import time
        
        delegate = tflite.load_delegate('libedgetpu.so.1')
        interpreter = tflite.Interpreter(
            model_path='mobilenet_edgetpu.tflite',
            experimental_delegates=[delegate]
        )
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        
        # Benchmark
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        fps = 1 / np.mean(times)
        
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Throughput: {fps:.2f} FPS")
        print(f"Min time: {np.min(times)*1000:.2f} ms")
        print(f"Max time: {np.max(times)*1000:.2f} ms")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

if __name__ == "__main__":
    print("=== Edge TPU Full Test ===")
    
    # Check device
    if not os.path.exists('/dev/apex_0'):
        print("✗ /dev/apex_0 not found")
        sys.exit(1)
    
    print("✓ /dev/apex_0 available")
    
    # Test inference
    if test_edgetpu_inference():
        print("\n🎉 SUCCESS: Edge TPU is fully operational!")
        
        # Run benchmark
        benchmark_edgetpu()
        
        print("\nYour Edge TPU is ready for:")
        print("• Image classification")
        print("• Object detection")
        print("• Running TensorFlow Lite models")
        print("• High-performance inference")
    else:
        print("\n❌ Edge TPU test failed")
