import warnings
# Suppress ONNX Runtime warning for Windows Server 2022
warnings.filterwarnings('ignore', category=UserWarning, message='.*Unsupported Windows version.*')

import onnxruntime as ort
import sys

def check_ort_gpu():
    print("-" * 50)
    print("ONNX Runtime GPU Diagnostic Tool")
    print("-" * 50)
    
    # 1. Print Python and ORT info
    print(f"Python version: {sys.version}")
    print(f"ONNX Runtime version: {ort.__version__}")
    
    # 2. Check installed providers
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    
    # 3. Check for CUDA
    if 'CUDAExecutionProvider' in providers:
        print("\n[SUCCESS] CUDAExecutionProvider is available!")
        print("ONNX Runtime is correctly linked with CUDA DLLs.")
        
        # Test if we can actually use it (optional, but get_available_providers is usually enough)
        try:
            device = ort.get_device()
            print(f"ONNX Runtime Device: {device}")
        except Exception as e:
            print(f"Note: Could not get device info: {e}")
    else:
        print("\n[WARNING] CUDAExecutionProvider is NOT available.")
        print("\nTo enable GPU support on Windows:")
        print("1. Uninstall onnxruntime:  pip uninstall onnxruntime")
        print("2. Install onnxruntime-gpu: pip install onnxruntime-gpu")
        print("3. Ensure CUDA and cuDNN are installed and added to your System PATH.")
    
    print("-" * 50)

if __name__ == "__main__":
    check_ort_gpu()
