import os
import json
import numpy as np
import tensorflow as tf
import download_model

from flask import Flask, render_template_string, request
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image


app = Flask(__name__)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_PATH = "archive/Train/Train"
TEST_PATH = "archive/test/test"

MODEL_PATH = "brain_tumor_model.keras"
CLASS_PATH = "class_names.json"

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "brain_tumor_model.keras not found."
    )

model = load_model("brain_tumor_model.keras")

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)


def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image = np.array(
        image
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        prediction
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = prediction[
        predicted_index
    ] * 100

    probabilities = []

    for i in range(
        len(class_names)
    ):

        probabilities.append({

            "class":
            class_names[i],

            "probability":
            round(
                float(
                    prediction[i] * 100
                ),
                2
            )

        })

    return (
        predicted_class,
        round(
            float(confidence),
            2
        ),
        probabilities
    )


HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
Brain Tumor MRI Prediction
</title>

<script src="https://cdn.tailwindcss.com">
</script>

</head>


<body class="bg-slate-100 min-h-screen">


<div class="max-w-4xl mx-auto px-6 py-12">


<div class="bg-white rounded-2xl shadow-xl p-8">


<h1 class="text-3xl font-bold text-center text-slate-800">

Brain Tumor MRI Classification

</h1>


<p class="text-center text-slate-500 mt-2">

Upload an MRI image to predict the class

</p>


<form

method="POST"

enctype="multipart/form-data"

class="mt-8"

>


<input

type="file"

name="image"

accept=".jpg,.jpeg,.png"

required

class="w-full border p-3 rounded-lg"

>


<button

type="submit"

class="w-full mt-5 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg"

>

Predict MRI

</button>


</form>


{% if error %}

<div class="mt-6 bg-red-100 text-red-700 p-4 rounded-lg">

{{ error }}

</div>

{% endif %}


{% if image_path %}

<div class="mt-8">


<h2 class="text-xl font-bold mb-4">

Uploaded MRI

</h2>


<img

src="{{ image_path }}"

class="w-full max-h-96 object-contain rounded-xl border"

>


</div>

{% endif %}


{% if prediction %}


<div class="mt-8 bg-slate-50 rounded-xl p-6">


<h2 class="text-2xl font-bold">

Prediction Result

</h2>


<div class="mt-5">


<p class="text-slate-600">

Predicted Class

</p>


<p class="text-3xl font-bold text-blue-600">

{{ prediction }}

</p>


</div>


<div class="mt-5">


<p class="text-slate-600">

Confidence

</p>


<p class="text-2xl font-bold text-green-600">

{{ confidence }}%

</p>


</div>


<div class="mt-8">


<h3 class="text-lg font-bold mb-4">

Class Probabilities

</h3>


{% for item in probabilities %}


<div class="mb-4">


<div class="flex justify-between mb-1">


<span>

{{ item.class }}

</span>


<span>

{{ item.probability }}%

</span>


</div>


<div class="w-full bg-slate-200 rounded-full h-3">


<div

class="bg-blue-600 h-3 rounded-full"

style="width: {{ item.probability }}%"

>

</div>


</div>


</div>


{% endfor %}


</div>


</div>


{% endif %}


</div>


<p class="text-center text-xs text-slate-500 mt-6">

For educational and research purposes only.
Not a medical diagnosis.

</p>


</div>


</body>

</html>

"""


@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    prediction = None

    confidence = None

    probabilities = None

    image_path = None

    error = None


    if request.method == "POST":


        if "image" not in request.files:

            error = "Please select an MRI image."

            return render_template_string(

                HTML,

                error=error

            )


        file = request.files["image"]


        if file.filename == "":

            error = "Please select an image."

            return render_template_string(

                HTML,

                error=error

            )


        filename = file.filename


        save_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            filename

        )


        file.save(
            save_path
        )


        try:

            (

                prediction,

                confidence,

                probabilities

            ) = predict_image(
                save_path
            )


            image_path = "/" + save_path.replace(
                "\\",
                "/"
            )


        except Exception as e:

            error = str(e)


    return render_template_string(

        HTML,

        prediction=prediction,

        confidence=confidence,

        probabilities=probabilities,

        image_path=image_path,

        error=error

    )


if __name__ == "__main__":
 
    app.run(
        debug=True
    )