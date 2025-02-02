#this code is where the dataset is split into train and validation sets

import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = r"C:\Users\jeferson fdo\PycharmProjects\Wheat Disease Detection\data\Dataset"
output_dir = r"C:\Users\jeferson fdo\PycharmProjects\Wheat Disease Detection\data\split_dataset"

train_dir = os.path.join(output_dir, "training data")
val_dir = os.path.join(output_dir, "validation data")

# Define train-validation split ratio
train_ratio = 0.8  # 80% train, 20% validation

# Create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through each class (folder)
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):  # Ensure it's a directory
        images = os.listdir(class_path)

        # Split images into train and validation sets
        train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

        # Create class folders inside train and validation directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Move images to respective folders
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print("✅ Dataset successfully split into training and validation sets.")