import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_PATH = "archive/Training"
TEST_PATH = "archive/Testing"

MODEL_PATH = "brain_tumor_model.keras"
CLASS_PATH = "class_names.json"

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)