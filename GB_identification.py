# %%
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

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

def convert_image_to_solid_circles_and_remove_dominant(x):
    image_path = f"Dataset/Grain_Boundary_Images/{x}.png"
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

        cv2.imwrite(f"Dataset/Grain_Boundary_Segmented/{x}.png", output_image)
        # cv2.imshow('Output Image', output_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("No circles detected.")

# Replace 'path_to_your_image.jpg' with the path to your image file
# convert_image_to_solid_circles_and_remove_dominant(x=36)

#the code till now processes the images and saves them in the folder Grain_Boundary_Segmented for further analysis

#working fine so far


# %%
#get the list of images to process
#go to the folder path and see the images presetn there, and check which ends by .png and add their name to a list
import os
path = "Dataset/Grain_Boundary_Images"
images_list = os.listdir(path)
images_list = [i.split('.')[0] for i in images_list if i.endswith(('.png', '.jpg', '.jpeg'))]

# Print the list of image names without their extensions
# print(images_list)

# %%
# first creat segmented images of all the normal images, seprating out the area of interest
for i in images_list:
    convert_image_to_solid_circles_and_remove_dominant(int(i))

# %%
import os
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

def create_augmented_images(image_path, augmentations_per_method=2):
    """
    Create augmented images from a single input image, saving multiple variations for each augmentation technique.

    :param image_path: Path to the original image.
    :param augmentations_per_method: Number of augmented images to create per method.
    """
    class_label = os.path.basename(image_path).split('.')[0]
    dir_name = f"Dataset/Augmented_dataset/{class_label}"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    image = Image.open(image_path)
    width, height = image.size

    # Augmentation operations
    augmentations = [
        ("rotate", lambda img, degree: img.rotate(degree)),
        ("scale", lambda img, factor: img.resize((int(width * factor), int(height * factor)))),
        ("translate", lambda img, shift: ImageOps.exif_transpose(img).transform(img.size, Image.AFFINE, (1, 0, shift, 0, 1, shift))),
        ("random_crop", lambda img: img.crop((np.random.randint(0, width // 4), np.random.randint(0, height // 4), width - np.random.randint(0, width // 4), height - np.random.randint(0, height // 4))))
    ]

    for aug_name, aug_func in augmentations:
        for i in range(augmentations_per_method):
            if aug_name == "rotate":
                degree = np.random.randint(-45, 45)  # Random rotation between -45 and 45 degrees
                augmented_image = aug_func(image, degree)
            elif aug_name == "scale":
                factor = np.random.uniform(0.8, 1.2)  # Random scaling between 80% and 120%
                augmented_image = aug_func(image, factor)
            elif aug_name == "translate":
                shift = np.random.randint(-width // 4, width // 4)  # Random translation
                augmented_image = aug_func(image, shift)
            elif aug_name == "random_crop":
                augmented_image = aug_func(image)  # Random cropping doesn't need parameters here

            augmented_image_path = os.path.join(dir_name, f"{class_label}_{aug_name}_{i}.png")
            augmented_image.save(augmented_image_path)

    #add the original image to the augmented images
    original_image_path = os.path.join(dir_name, f"{class_label}_original.png")
    image.save(original_image_path)

#augmenting all the images and saving them to a directory
for image in images_list:
    create_augmented_images(f"Dataset/Grain_Boundary_Segmented/{image}.png")


# %%
import os
import random
from itertools import combinations, product

def create_and_shuffle_pairs(dataset_dir):
    # List all class directories
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    similar_pairs = []
    dissimilar_pairs = []
    
    # Generate similar pairs (within the same class)
    for class_dir in class_dirs:
        image_files = os.listdir(os.path.join(dataset_dir, class_dir))
        image_paths = [os.path.join(dataset_dir, class_dir, img) for img in image_files]
        # Create all possible combinations of 2 images within the same class
        for img1, img2 in combinations(image_paths, 2):
            similar_pairs.append((img1, img2, 1))
    
    # Generate dissimilar pairs (across different classes)
    for class1, class2 in combinations(class_dirs, 2):
        image_files1 = os.listdir(os.path.join(dataset_dir, class1))
        image_files2 = os.listdir(os.path.join(dataset_dir, class2))
        image_paths1 = [os.path.join(dataset_dir, class1, img) for img in image_files1]
        image_paths2 = [os.path.join(dataset_dir, class2, img) for img in image_files2]
        # Create pairs from one image in class1 and another in class2
        for img1, img2 in product(image_paths1, image_paths2):
            dissimilar_pairs.append((img1, img2, 0))
    
    # Shuffle both sets of pairs to ensure randomness
    random.shuffle(similar_pairs)
    random.shuffle(dissimilar_pairs)
    
    # Now, to ensure there's no bias due to order or proportion, you can balance the dataset here
    # For example, you might want to sample an equal number of similar and dissimilar pairs
    # This step depends on the specific needs of your dataset and model.
    # Assuming you want to keep all dissimilar pairs and a matching number of similar pairs:
    min_pairs = min(len(similar_pairs), len(dissimilar_pairs))
    balanced_similar_pairs = random.sample(similar_pairs, min_pairs)
    balanced_dissimilar_pairs = random.sample(dissimilar_pairs, min_pairs)

    # Combine and shuffle again to mix similar and dissimilar pairs
    combined_pairs = balanced_similar_pairs + balanced_dissimilar_pairs
    random.shuffle(combined_pairs)

    print(len(balanced_similar_pairs), len(balanced_dissimilar_pairs))
    
    return combined_pairs

# Example usage
dataset_dir = 'Dataset/Augmented_dataset'  # Path to the dataset directory
combined_pairs = create_and_shuffle_pairs(dataset_dir)
print(f"Total pairs after balancing and shuffling: {len(combined_pairs)}")


# %%
print(combined_pairs)

# %%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model

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


def build_base_network(input_shape):
    base_network = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
    ])
    return base_network

def build_siamese_model(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_network = build_base_network(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Use the custom layer instead of a Lambda layer
    distance = AbsoluteDifferenceLayer()([processed_a, processed_b])

    # Add a dense layer with a single neuron and a sigmoid activation to generate the similarity score
    outputs = Dense(1, activation='sigmoid')(distance)

    model = Model(inputs=[input_a, input_b], outputs=outputs)
    return model


# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_path, target_size=(224, 224)):
    img = keras_image.load_img(image_path, target_size=target_size)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img[0]

def load_and_process_pairs(pairs, target_size=(224, 224)):
    # Splitting image paths and labels
    image_paths_a, image_paths_b, labels = zip(*pairs)

    # Preprocessing images
    images_a = np.array([preprocess_image(path, target_size) for path in image_paths_a])
    images_b = np.array([preprocess_image(path, target_size) for path in image_paths_b])
    
    return images_a, images_b, np.array(labels)

# Split the dataset into training and validation sets
pairs_train, pairs_val = train_test_split(combined_pairs, test_size=0.2, random_state=42)

# Load and preprocess images for training and validation
images_a_train, images_b_train, labels_train = load_and_process_pairs(pairs_train)
images_a_val, images_b_val, labels_val = load_and_process_pairs(pairs_val)


# %%
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Assuming `build_siamese_model` is a function you have defined to build your 
input_shape = (224, 224,3)
model = build_siamese_model(input_shape)

# Compile the model with the corrected Adam optimizer reference
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Then proceed with your training as planned
model.fit([images_a_train, images_b_train], labels_train,
          validation_data=([images_a_val, images_b_val], labels_val),
          batch_size=32, epochs=1)


# %%


# %%
# Assuming your model is named `model` and you've built it with the custom layer correctly implemented
model_path = 'my_siamese_model.keras'

# Save the model
model.save(model_path)

# # To load the model, specify custom objects if necessary
# from tensorflow.keras.models import load_model

# # If you used custom layers or other custom components, specify them in custom_objects
# loaded_model = load_model(model_path, custom_objects={'AbsoluteDifferenceLayer': AbsoluteDifferenceLayer})


# %%
# To load the model, specify custom objects if necessary
from tensorflow.keras.models import load_model

# If you used custom layers or other custom components, specify them in custom_objects
model = load_model("my_siamese_model.keras", custom_objects={'AbsoluteDifferenceLayer': AbsoluteDifferenceLayer})

# %%
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

# %%
import os
import numpy as np

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

# Example usage
test_image_path = 'Test_Image/40.png'
reference_images_dir = 'Dataset/Grain_Boundary_Segmented'
predicted_class, similarity_scores = classify_test_image(test_image_path, reference_images_dir, model)
print(f"Predicted Class: {predicted_class}")
print("Similarity Scores:", similarity_scores)


