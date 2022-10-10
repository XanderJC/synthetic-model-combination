import torch

try:
    if torch.backends.mps.is_available() & torch.backends.mps.is_built():
        DEVICE = "mps"
    else:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if DEVICE == "mps":
    print("Using Apple Silicon GPU")
elif DEVICE == "cuda:0":
    print("Using NVIDIA GPU")
else:
    print("No GPU detected -> Using CPU")
