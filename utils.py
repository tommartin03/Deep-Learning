import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import requests
from IPython.core.display import display, HTML

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(base_dir, split):
    """
    Load images and labels from the specified split (train, val, or test).
    
    Args:
        split (str): The split type, one of 'train', 'val', or 'test'.
        base_dir (str): The base directory containing images folder and txt files.
        
    Returns:
        images (list of PIL Images): List of loaded images.
        labels (list of int or None): List of labels (if split is 'train' or 'val') or None (if split is 'test').
    """
    images = []
    labels = []
    
    if split == 'test':
        # Load image paths from test_without_labels.txt for test split
        txt_file = os.path.join(base_dir, 'test_without_labels.txt')
        with open(txt_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        
        # Load images without labels
        for image_path in image_paths:
            img = Image.open(image_path)
            img = img.convert("L")
            images.append(img)
            labels.append(-1)
            
    else:
        # Load image paths and labels for train or val split
        txt_file = os.path.join(base_dir, f'{split}.txt')
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        # Load images and labels
        for line in lines:
            image_path, label = line.strip().split()
            img = Image.open(image_path)
            img = img.convert("L")
            images.append(img)
            labels.append(int(label))
        # Conversion de X et y en tableaux numpy
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def show_image(image_array, image_class=0):
    # Affichage de l'image
    plt.imshow(image_array, cmap="gray")
    plt.title(f"Image MNIST : {image_class}")
    plt.axis("off")
    plt.show()

    # Affichage du tableau des pixels
    print("Matrice de pixels :")
    print(image_array)

def get_result(name, pred_file = 'predictions.csv'):
    # Define the URL and the files you want to send
    url = "http://webvinc.iuto.ovh/evaluate"
    files = {'file': open(pred_file, 'rb')}
    data = {'name': name}
    # Make the POST request
    response = requests.post(url, files=files, data=data)
    # Print the response
    result = response.json()

    try:
        print(f"Accuracy = {result['evaluation']['accuracy']}")
        print("------------------DETAILS-------------------")
        print(f"{'Class':<6} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Support':<10}")
        # Extract and print each class's metrics
        for key, metrics in result['evaluation'].items():
            if key != 'accuracy':  # Exclude the accuracy key
                print(f"{key:<6} {metrics['f1-score']:<10.3f} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['support']:<10.1f}")
    except:
        print (result)
    return response

def show_ranking():
    # Step 1: Call the API
    url = "http://webvinc.iuto.ovh/ranking"  # Replace with your API endpoint
    response = requests.get(url)  # Use requests.post() if it's a POST request
    # Step 2: Check if the request was successful
    if response.status_code == 200:
        # Step 3: Display the HTML content
        display(HTML(response.text))
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Fonction pour valider le modèle sur les données de validation
def validate_model(model, val_loader):
    model.eval()  # Mode évaluation (pas de mise à jour des poids)
    correct = 0
    total = 0
    with torch.no_grad():  # Pas de calcul des gradients
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total * 100
    print(f"Validation Accuracy: {val_acc:.2f}%")

import torch
from torch.utils.data import DataLoader
import numpy as np


def predict_test(model, test_loader):
    model.eval()
    predictions = []

    # Loop through the test data and make predictions
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # Make predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions
            predictions.extend(predicted.cpu().numpy())
    return predictions

import random
def random_predict_and_show(model, test_loader):

    # Pick a random index from the DataLoader
    random_idx = random.randint(0, len(test_loader) - 1)
    # Get the batch at the random index
    for idx, (inputs, labels) in enumerate(test_loader):
        if idx == random_idx:
            break
    image = inputs[0]
    model.eval()  
    with torch.no_grad(): 
        output = model(image) 
        if output.dim() == 1:
        	output = output.unsqueeze(0) 
        _, predicted = torch.max(output, 1)

    image = image.squeeze()  
    prediction = predicted.item()

    plt.imshow(image.reshape(28,28).numpy(), cmap='gray')
    plt.title(f"Predicted: {prediction}") 
    plt.show()
