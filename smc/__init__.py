import torch

try:
    if torch.backends.mps.is_available() & torch.backends.mps.is_built():  # type: ignore
        DEVICE = "mps"
    else:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:  # pylint: disable=broad-except
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if "mps" in DEVICE.type:  # type: ignore
    print("Using Apple Silicon GPU")
elif "cuda" in DEVICE.type:  # type: ignore
    print("Using NVIDIA GPU")
else:
    print("No GPU detected -> Using CPU")
