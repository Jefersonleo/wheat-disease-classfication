import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define the path to your trained model and load it
model_path = r'C:\Users\jeferson fdo\PycharmProjects\Wheat Disease Detection\wheat_disease_resnet18.pth'
model = models.resnet18(pretrained=False)  # Example if you're using ResNet18
num_features = 5  # As per your training, the number of classes

# Replace the final layer to match the number of classes in your model
model.fc = nn.Linear(model.fc.in_features, num_features)

# Load the trained model weights
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the model's input size
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats (change if needed)
])


class_names = ['Brown_rust', 'Healthy', 'loose smut', 'septoria', 'Yellow rust']

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open and transform the image
        image = Image.open(file.stream)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)  # Get the class with the highest probability

        # Get the predicted class label
        predicted_class_name = class_names[predicted_class.item()]
        return jsonify({'predicted_class': predicted_class_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')
# Run the app
if __name__ == '__main__':
    app.run(debug=True)