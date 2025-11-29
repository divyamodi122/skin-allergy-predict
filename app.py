from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import uuid

# Load trained CNN model
model = tf.keras.models.load_model("skin_cnn_model.keras")

# Flask app
app = Flask(__name__)

# Class names and medicine dictionary
class_names = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis",
    "Psoriasis",
    "Seborrheic Keratoses",
    "Fungal Infection"
]

medicine = {
    "Eczema": "Hydrocortisone cream, Moisturizer, Avoid hot showers",
    "Melanoma": "Immediate dermatologist referral needed, No self-medicine",
    "Atopic Dermatitis": "Cetaphil lotion, Mild steroid cream",
    "Basal Cell Carcinoma": "Visit dermatologist, Surgical removal needed",
    "Melanocytic Nevi": "Usually harmless, monitor changes",
    "Benign Keratosis": "AHA creams, Dermatologist liquid nitrogen",
    "Psoriasis": "Coal tar, Salicylic acid shampoo, Moisturizing creams",
    "Seborrheic Keratoses": "AHA creams, Cryotherapy by dermatologist",
    "Fungal Infection": "Clotrimazole cream, Ketoconazole soap"
}

# Routes
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return "No file uploaded!"

    # Save uploaded file in static folder
    upload_folder = os.path.join(app.root_path, "static")
    os.makedirs(upload_folder, exist_ok=True)
    filename = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    # Preprocess image for model
    img = image.load_img(file_path, target_size=(180, 180))  # same size as model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    index_pred = np.argmax(prediction)
    disease = class_names[index_pred]
    med = medicine[disease]

    # Render result page
    return render_template("result.html",
                           disease=disease,
                           medicine=med,
                           img=filename)

# Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222, debug=True)
