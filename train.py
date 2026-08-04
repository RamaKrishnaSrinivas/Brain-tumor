import os
import json
import numpy as np

import tensorflow as tf
from config import (
    IMG_SIZE,
    BATCH_SIZE,
    TRAIN_PATH,
    TEST_PATH,
    MODEL_PATH,
    CLASS_PATH,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
def train_model():

    print("Training model...")

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(
        rescale=1./255
    )

    train_data = train_gen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_gen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print("Class indices:")
    print(train_data.class_indices)

    class_names = list(train_data.class_indices.keys())

    with open(CLASS_PATH, "w") as f:
        json.dump(class_names, f)

    model = Sequential([

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(224, 224, 3)
        ),

        MaxPooling2D(2, 2),

        Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        MaxPooling2D(2, 2),

        Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        MaxPooling2D(2, 2),

        Flatten(),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            len(class_names),
            activation="softmax"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=10
    )

    loss, accuracy = model.evaluate(test_data)

    print(
        "Test Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    model.save(MODEL_PATH)

    print("Model saved successfully.")

    return model, class_names

model, class_names = train_model()