import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained model
model = load_model('creative_classifier_model.h5')  # Replace with the actual path to your model

# Define the target image size expected by the model
img_size = (256, 256)

def preprocess_image(image):
    # Resize the image to the target size and convert to array
    img = image.resize(img_size)
    img_array = np.asarray(img)
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_creativity(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

def main():
    st.title("Creative Image Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make a prediction
        image = Image.open(uploaded_file)
        creativity_score = predict_creativity(image)

        # Display the prediction result
        st.write("Prediction Score:", creativity_score)

        # Determine and display the classification result
        if creativity_score > 0.5:
            st.write("This image is classified as creative.")
        else:
            st.write("This image is classified as non-creative.")

if __name__ == "__main__":
    main()
