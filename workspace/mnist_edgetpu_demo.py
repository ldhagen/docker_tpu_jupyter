import tensorflow as tf
import numpy as np
from edgetpu_utils import EdgeTPUHelper
import os

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and preprocess
X_train, X_test = X_train / 255.0, X_test / 255.0

# Create an Edge TPU compatible model
def create_edgetpu_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Create and train model
model = create_edgetpu_mnist_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training MNIST model...")
model.fit(X_train, y_train, epochs=3, validation_split=0.1)  # Reduced epochs for testing

# Convert to TensorFlow Lite for Edge TPU
def convert_to_edgetpu_tflite(keras_model, representative_data):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset():
        for i in range(100):
            # Ensure the data has the right shape (1, 28, 28, 1)
            yield [representative_data[i:i+1].reshape(1, 28, 28, 1).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    return tflite_model

# Convert model
print("Converting model for Edge TPU...")
tflite_model = convert_to_edgetpu_tflite(model, X_train)

# Save the converted model
with open('mnist_edgetpu.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as 'mnist_edgetpu.tflite'")

# Test with Edge TPU
edgetpu_helper = EdgeTPUHelper()
if edgetpu_helper.load_model('mnist_edgetpu.tflite'):
    print("\n🧪 Testing Edge TPU inference...")
    
    # Get model info to understand expected input
    model_info = edgetpu_helper.get_model_info()
    print(f"Model expects input: {model_info['input_shape']} with dtype: {model_info['input_dtype']}")
    
    # Test with a few samples
    for i in range(5):
        sample = X_test[i]
        true_label = y_test[i]
        
        # Prepare input for Edge TPU (quantize to uint8)
        sample_input = (sample * 255).astype(np.uint8)
        
        print(f"Sample {i} shape before reshaping: {sample_input.shape}")
        
        # Run inference - let the helper handle reshaping
        prediction, inference_time = edgetpu_helper.predict(sample_input)
        predicted_label = np.argmax(prediction)
        
        print(f"Sample {i}: True={true_label}, Predicted={predicted_label}, "
              f"Time={inference_time*1000:.2f}ms, Correct={true_label==predicted_label}")
        print(f"Raw prediction: {prediction.flatten()}")
    
    # Benchmark performance
    test_sample = (X_test[0] * 255).astype(np.uint8)
    benchmark = edgetpu_helper.benchmark(test_sample, num_runs=50)
    
    print(f"\n📊 Edge TPU Performance:")
    print(f"Average inference time: {benchmark['mean_time']*1000:.2f}ms")
    print(f"Throughput: {benchmark['fps']:.2f} FPS")
    
else:
    print("Failed to load Edge TPU model")
