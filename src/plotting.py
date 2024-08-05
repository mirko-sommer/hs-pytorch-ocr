import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from utils import decode_logits



def plot_loss(epoch: int, train_losses: list, val_losses: list, n_steps: int = 100):
    """
    Plots train and validation losses.

    Args:
        epoch (int): Current epoch number.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        n_steps (int, optional): Number of most recent steps to average for the title. Default is 100.
    """
    # Clear previous graph
    clear_output(True)

    # Making titles
    train_title = f'Epoch: {epoch} | Training Loss'
    val_title = f'Epoch: {epoch} | Validation Loss'

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

    # Plot training losses
    ax[0].plot(train_losses, color='mediumblue', label='Training Loss', linewidth=2)
    ax[0].set_title(train_title, fontsize=14)
    ax[0].set_xlabel('Iterations', fontsize=12)
    ax[0].set_ylabel('Loss', fontsize=12)
    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[0].legend()

    # Plot validation losses
    ax[1].plot(val_losses, color='darkorchid', label='Validation Loss', linewidth=2)
    ax[1].set_title(val_title, fontsize=14)
    ax[1].set_xlabel('Iterations', fontsize=12)
    ax[1].set_ylabel('Loss', fontsize=12)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def print_sample_image(model, dataset, device, label_encoder):
    """
    Displays a sample image with its predicted and actual text.

    Args:
        model: The PyTorch model used for inference.
        dataset: The dataset from which to sample images.
        label_encoder: The encoder used to decode model predictions.
        device: The device on which to run the model (CPU or GPU).
    """
    idx = np.random.randint(len(dataset))
    
    img, target_text = dataset[idx]
    
    img_input = img.unsqueeze(0).to(device)  # Add batch dimension
    logits = model(img_input)
    pred_text = decode_logits(logits.cpu(), label_encoder)
    
    # Convert image from (C, H, W) to (H, W, C) and plot
    img = img.permute(1, 2, 0).numpy()
    
    title = f'Gold: {target_text} \ Pred: {pred_text}'
    plt.imshow(img)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.show()


def print_sample_grid(model, dataset, label_encoder, device):
    """
    Displays a grid of sample images with their predicted and actual texts.

    Args:
        model: The PyTorch model used for inference.
        dataset: The dataset from which to sample images.
        label_encoder: The encoder used to decode model predictions.
        device: The device on which to run the model (CPU or GPU).
    """

    # Randomly select `num_samples` samples
    indices = np.random.choice(len(dataset), 9, replace=False)
    fig, axs = plt.subplots(3, 3, figsize=(8, 4))
    axs = axs.flatten()

    for ax, idx in zip(axs, indices):
        img, target_text = dataset[idx]
        logits = model(img.unsqueeze(0).to(device))
        pred_text = decode_logits(logits.cpu(), label_encoder)

        img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        img = img.numpy()

        title = f'Gold: {target_text}\nPred: {pred_text}'
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
