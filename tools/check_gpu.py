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
        try:
            # Try to create a dummy session to verify CUDA
            # This requires a very small dummy model or just testing the device
            print("Attempting to initialize a session with CUDA...")
            # We don't necessarily need a model file to check if ORT can talk to CUDA
            # but testing if it's in the list is usually enough to know the DLLs are there.
            session = ort.InferenceSession(None, providers=['CUDAExecutionProvider'])
        except Exception as e:
            if "InferenceSession.__init__() takes 2 positional arguments but 3 were given" in str(e):
                # Older versions of ORT
                pass
            elif "None is not a valid" in str(e):
                # This is expected since we passed None for model
                print("CUDA DLLs seem to be correctly loaded.")
            else:
                print(f"\n[ERROR] Failed to initialize CUDA session: {e}")
                print("\nPossible reasons:")
                print("1. CUDA version mismatch (e.g., ORT expects CUDA 11.x but 12.x is installed).")
                print("2. cuDNN DLLs (cudnn64_8.dll etc.) are missing from PATH.")
                print("3. zlibwapi.dll is missing (required by some cuDNN versions).")
    else:
        print("\n[WARNING] CUDAExecutionProvider is NOT available.")
        print("\nTo enable GPU support on Windows:")
        print("1. Uninstall onnxruntime:  pip uninstall onnxruntime")
        print("2. Install onnxruntime-gpu: pip install onnxruntime-gpu")
        print("3. Ensure CUDA and cuDNN are installed and added to your System PATH.")
    
    print("-" * 50)

if __name__ == "__main__":
    check_ort_gpu()
