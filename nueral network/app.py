import torch
import gradio as gr
import torchvision.transforms as transforms
from torchvision import datasets
import random
import numpy as np

# Load the model structure
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 128)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128, 64)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the model and weights
model = NeuralNetwork()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load MNIST test dataset
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

# Select 10-15 random images
def get_random_images():
    indices = random.sample(range(len(test_dataset)), 15)
    images = [(test_dataset[i][0], test_dataset[i][1]) for i in indices]  # Store image and label pairs
    return images

# Prediction function
def predict_image(img):
    image_tensor = transforms.ToTensor()(img).view(1, 28*28)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    
    return img, f"Predicted Label: {predicted_label}"

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# 🧠 MNIST Digit Classifier")
    gr.Markdown("### Select an image to test the model!")

    images = get_random_images()
    gallery = gr.Gallery([image[0].numpy().squeeze() for image in images], label="Select an Image", columns=5, height=100)
    output_image = gr.Image(label="Selected Image")
    output_text = gr.Textbox(label="Prediction Result")

    gallery.select(predict_image, inputs=[gallery], outputs=[output_image, output_text])

app.launch()
