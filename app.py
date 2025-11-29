from flask import Flask, render_template, request           #flask ko import kiya webapp banan k liye
import tensorflow as tf                                #model load krne k liye
import numpy as np                                     #array processing
from tensorflow.keras.preprocessing import image            #image module image load kliye
import os                                           #os file path banane k liye

# Load model
model = tf.keras.models.load_model("skin_cnn_model.keras")

app = Flask(__name__)                              #flask application ka object bannata hai

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

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])                   #user image upload karta hai to prdict button se request post s yha ati hainimage nd run hoti hai
def predict():
    file = request.files.get("file")                #form se jo image bhji gye hai uspe access 

    if file is None or file.filename == "":
        return "No file uploaded!"

    # Save file
    upload_folder = os.path.join(app.root_path, "static")
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, "uploaded.jpg")
    file.save(file_path)

    # Preprocessing--- image upload krta hai ,size fix 180*180 main, numpt array main convert krta nd normalize krta 0-255 to 0-1
    img = image.load_img(file_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction- - model disease ka probabilty output deta hai
    prediction = model.predict(img_array)
    index_pred = np.argmax(prediction)                 #argmax se max probability index milta hai
    disease = class_names[index_pred]                 #index k hisab se diseases ka nm milta nd dictionry s medicine milti hai
    med = medicine[disease]

    return render_template("result.html",            
                           disease=disease,
                           medicine=med,
                           img="uploaded.jpg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2222, debug=True)

#render_templates -- html file show krta hai [flask python code ko html se connect karta hai]
#request --- user se input leta hai(file,form,button)
#template variable -- python se html me data bhjeta hain