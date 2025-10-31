#!/usr/bin/env python3
"""
Real-world Edge TPU example with image classification
"""
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import numpy as np

# Load model
interpreter = make_interpreter('mobilenet_v1_1.0_224_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# Get input size
width, height = common.input_size(interpreter)

# Create a test image (or load a real one)
image = Image.new('RGB', (width, height), color='red')
image = image.convert('RGB')  # Ensure RGB format

# Preprocess image
image = image.resize((width, height), Image.Resampling.LANCZOS)
input_data = np.array(image, dtype=np.uint8)

# Run inference
common.set_input(interpreter, input_data)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=3)

print("Top 3 predictions:")
for c in classes:
    print(f"  Class {c.id}: {c.score:.4f}")

print("🎉 Real example working!")
