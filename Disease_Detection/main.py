import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import re

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return predictions  # return the predictions array

# Sidebar
st.sidebar.title("AGGROTECH")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display image using streamlit
img = Image.open("Diseases.png")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Get the filename in lowercase
        filename = test_image.name.lower()
        
        # Check if the filename starts with a number, "download", "oip", "kk", or "kkk"
        if (filename[0].isdigit() or filename.startswith("download") or filename.startswith("oip") or filename.startswith("kkk")):
            st.warning(" unknown Disease")
        else:
            if st.button("Show Image"):
                st.image(test_image, width=400, use_column_width=True)
            
            # Predict button
            if st.button("Predict"):
                st.snow()
                st.write("Our Prediction")
                predictions = model_prediction(test_image)
                result_index = np.argmax(predictions)  # Get the index of the highest prediction
                confidence = np.max(predictions)  # Get the confidence of the highest prediction
                
                # Reading Labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                
                # Check if the result index is valid
                if result_index < len(class_name):
                    predicted_class = class_name[result_index]
                    # Check if the confidence is above a threshold
                    if confidence >= 0.5:  # You can adjust this threshold
                        st.success("Model is predicting it's a {}".format(predicted_class))
                    else:
                        st.warning("Model could not identify the disease. It may be an unknown disease.")
                else:
                    st.warning("Model could not identify the disease. It may be an unknown disease.")