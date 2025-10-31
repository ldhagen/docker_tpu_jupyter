#!/usr/bin/env python3
# Quick Edge TPU test script with error handling
import time
import sys

def test_edgetpu():
    for attempt in range(3):
        try:
            from pycoral.utils.edgetpu import list_edge_tpus
            import tflite_runtime.interpreter as tflite
            
            print("✓ Pycoral imports working")
            
            devices = list_edge_tpus()
            print(f"✓ Found {len(devices)} Edge TPU devices")
            
            delegate = tflite.load_delegate('libedgetpu.so.1')
            print("✓ Edge TPU delegate loaded")
            
            print("🎉 All tests passed!")
            return True
            
        except Exception as e:
            if attempt < 2:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                print("   Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"❌ All attempts failed: {e}")
                return False
    return False

if __name__ == "__main__":
    success = test_edgetpu()
    sys.exit(0 if success else 1)
