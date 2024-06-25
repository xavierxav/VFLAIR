import os
import random
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
tp = transforms.ToTensor()

def set_seed(seed=0):
    """
    Set the seed for reproducibility across various libraries.
    
    Parameters:
    seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_size_of(target_tensor):
    total_bytes = target_tensor.numel() * target_tensor.element_size()
    return total_bytes*4/(1024*1024) # MB

def plot_model_performance(model_names, accuracy):
    """
    Plots the model performance by epochs with min-max accuracy as a light cone and mean accuracy as a darker line.
    
    Parameters:
    - accuracy: numpy array of shape (num_models, num_seeds, num_epochs)
      The accuracy values for different models, seeds, and epochs.
    - model_names: list of strings
      The names of the models.
    """
    num_models, num_seeds, num_epochs = accuracy.shape
    epochs = np.arange(1, num_epochs + 1)
    
    # Define colors for different models
    colors = plt.cm.get_cmap('tab10', num_models)
    
    plt.figure(figsize=(12, 8))
    
    for model_index in range(num_models):
        model_accuracies = accuracy[model_index, :, :]
        
        # Compute mean, min, and max for each epoch
        mean_accuracy = model_accuracies.mean(axis=0)
        min_accuracy = model_accuracies.min(axis=0)
        max_accuracy = model_accuracies.max(axis=0)
        
        # Plot the light cone (min-max range)
        plt.fill_between(epochs, min_accuracy, max_accuracy, color=colors(model_index), alpha=0.15)
        
        # Plot the mean accuracy
        plt.plot(epochs, mean_accuracy, color=colors(model_index), label=f'{model_names[model_index]} Mean Accuracy')
    
    # Add titles and labels
    plt.title('Model Performance by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.show()
