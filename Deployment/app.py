import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import time

def energy(pid):
    pid = int(pid)
    # Load the DataFrame
    df = pd.read_csv("Numerical_Data/energy_ni.csv")
    df['GB_index'] = df['GB_index'].astype(int)
    energy = df.loc[df['GB_index'] == pid, 'Ni(J/m^2)'].values[0]
    return energy

class AbsoluteDifferenceLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Assumes inputs is a list of two tensors of the same shape.
        return tf.abs(inputs[0] - inputs[1])

    def get_config(self):
        # This method is used for serialization.
        base_config = super().get_config()
        return base_config

def save_uploaded_file(uploaded_file):
    """ Save the uploaded file to a directory. """
    try:
        os.makedirs('Deployment/saved_images', exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join('Deployment/saved_images', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Write file to disk
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def preprocess_image(image_path, target_size=(224, 224)):
    img = keras_image.load_img(image_path, target_size=target_size)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img[0]

def calculate_average_color(image, center, radius):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    mean_val = cv2.mean(image, mask=mask)
    return (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))

def find_dominant_color(colors):
    # Using KMeans clustering to find clusters of colors
    if len(colors) > 1: # Need at least 2 colors to cluster
        kmeans = KMeans(n_clusters=min(5, len(colors)), random_state=0).fit(colors)
        counter = Counter(kmeans.labels_)
        most_common = counter.most_common(1)[0][0]
        dominant_color = kmeans.cluster_centers_[most_common]
    else:
        dominant_color = colors[0] if colors else (0, 0, 0)
    return dominant_color

def is_color_similar(color1, color2, threshold=45):
    # Calculate Euclidean distance between two colors
    distance = np.sqrt(sum([(a - b) ** 2 for a, b in zip(color1, color2)]))
    return distance < threshold

def convert_test_image_to_segmented(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=20, minRadius=8, maxRadius=12)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        colors = [calculate_average_color(image, (i[0], i[1]), i[2]) for i in circles[0, :]]
        dominant_color = find_dominant_color(colors)

        output_image = np.zeros_like(image)
        for i, color in zip(circles[0, :], colors):
            if not is_color_similar(color, dominant_color):
                cv2.circle(output_image, (i[0], i[1]), i[2], color, -1)

        #save this image at directory
        cv2.imwrite(f"Test_Image/Misc/test_image.png", output_image)

def classify_test_image(test_image_path, reference_images_dir, model, target_size=(224, 224)):
    """
    Classify a test image by comparing it with reference images from each class.
    
    :param test_image_path: Path to the test image.
    :param reference_images_dir: Directory containing reference images named after their class.
    :param model: Trained Siamese network model.
    :param target_size: Target size for resizing images before processing.
    :return: Predicted class name for the test image.
    """
    convert_test_image_to_segmented(test_image_path)
    test_image_path = "Test_Image/Misc/test_image.png"
    test_image_preprocessed = preprocess_image(test_image_path, target_size)
    similarity_scores = {}

    for reference_image_name in os.listdir(reference_images_dir):
        if reference_image_name.endswith('.png'):
            class_name = os.path.splitext(reference_image_name)[0]
            reference_image_path = os.path.join(reference_images_dir, reference_image_name)
            reference_image_preprocessed = preprocess_image(reference_image_path, target_size)
            
            # Use the trained Siamese model to compute similarity
            similarity = model.predict([np.expand_dims(test_image_preprocessed, axis=0), np.expand_dims(reference_image_preprocessed, axis=0)])[0][0]
            similarity_scores[class_name] = similarity

    # The class with the highest similarity score is considered the predicted class
    predicted_class = max(similarity_scores, key=similarity_scores.get)
    #delete all the content in misc
    for i in os.listdir("Test_Image/Misc"):
        os.remove(f"Test_Image/Misc/{i}")
    return predicted_class, similarity_scores

def pid_to_boundary(pid):
    pid_mapped_dict = np.load('Numerical_Data/pid_mapped_dict.npy', allow_pickle=True).item()
    return pid_mapped_dict[pid]

def title_show():
    # Title and Introduction
    st.title("Grain Boundary Classification for Bicrystalline Materials using Molecular Dynamic simulation images")
    st.write("""
    ## One-Shot Learning with Siamese Networks
    This application demonstrates the use of one-shot learning techniques to classify grain boundaries in bicrystalline materials. By leveraging the capabilities of Siamese neural networks, this tool can effectively predict grain boundary types from a single example per class.
    
    ### How It Works
    - **Upload an Image**: Provide an image of a bicrystalline material's grain boundary MD simulation.
    - **Model Prediction**: Our pre-trained Siamese model analyzes the image to classify the type of grain boundary.
    - **Results**: The app displays the predicted grain boundary type, the confidence of the prediction, and relevant energy values associated with the boundary.
    
    ### Why One-Shot Learning?
    In many scientific fields, especially materials science, obtaining large datasets of labeled images can be costly and time-consuming. One-shot learning allows us to make accurate predictions with limited data, focusing on learning from few examples rather than from a large dataset.
    
    ### Future Directions
    #### Application in Polycrystalline Materials
    The technology behind this application is currently being expanded to include Grain-Boundary recognition capabilities. This advancement will allow the app to not only classify but also identify and locate various types of grain boundaries within polycrystalline materials. This extension aims to provide a more comprehensive analysis tool that can assist in the detailed study of GB structures.
    
    **Please upload an image of a bicrystalline material below to see the model in action.**
    """)

def main():
    title_show()
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            st.success("File Saved Successfully")
            file_path = os.path.join("Deployment/saved_images", uploaded_file.name)
            st.write(f"Saved file path: {file_path}")

            print(file_path)
            
            # Display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Load model with custom objects
            model = load_model("Models/my_siamese_model.keras", custom_objects={'AbsoluteDifferenceLayer': AbsoluteDifferenceLayer})
            
            # Define the directory containing reference images
            reference_images_dir = 'Dataset/Grain_Boundary_Segmented'
            with st.spinner('Processing... Please wait.'):
                # Perform classification
                predicted_class, similarity_scores = classify_test_image(file_path, reference_images_dir, model)
            # Using st.metric to display key results
            st.metric(label="Predicted PID", value=predicted_class)
            st.metric(label="Predicted Boundary", value=pid_to_boundary(predicted_class))
            st.metric(label="Energy (J/m²)", value=f"{energy(predicted_class):.3f}")

            # Expander for additional details
            with st.expander("In case of any ambiguity, you can look at the confidence scores:"):
                scores_df = pd.DataFrame(list(similarity_scores.items()), columns=['PID Class', 'Confidence Score'])
                st.dataframe(scores_df.sort_values('Confidence Score', ascending=False))
        else:
            st.error("Failed to save file")

if __name__ == "__main__":
    main()
