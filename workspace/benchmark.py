#!/usr/bin/env python3
import time
from pycoral.utils.edgetpu import make_interpreter
import numpy as np

interpreter = make_interpreter('mobilenet_v1_1.0_224_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# Warm up
input_shape = interpreter.get_input_details()[0]['shape']
input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()

# Benchmark
times = []
for i in range(100):
    start = time.time()
    interpreter.invoke()
    end = time.time()
    times.append((end - start) * 1000)  # Convert to milliseconds

avg_time = np.mean(times)
print(f"Average inference time: {avg_time:.2f}ms")
print(f"Throughput: {1000/avg_time:.1f} FPS")
