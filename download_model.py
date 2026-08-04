import os
import urllib.request

MODEL_URL = "https://huggingface.co/rkso/brain-tumor-model/resolve/main/brain_tumor_model.keras"
MODEL_PATH = "brain_tumor_model.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")