import os
import shutil
import random
import torch

from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple
from PIL import Image


def is_empty(directory: Path) -> bool:
    """Check if the directory is empty."""
    return not any(directory.iterdir())

def split_dataset(src_path: Path, train_path: Path, test_path: Path, val_path: Path, 
                  train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, 
                  iam=False, num_samples=None):
    """
    Splits a directory with image files into training, test, and validation sets, if target directories are empty.
    Selects a fixed number of samples for splitting if num_samples is provided.

    Args:
        src_path (Path): Source directory containing image files.
        train_path (Path): Target directory for training data.
        test_path (Path): Target directory for test data.
        val_path (Path): Target directory for validation data.
        train_ratio (float): Proportion of training data (between 0 and 1).
        test_ratio (float): Proportion of test data (between 0 and 1).
        val_ratio (float): Proportion of validation data (between 0 and 1).
        iam (bool): Whether to preserve the subdirectory structure for IAM dataset.
        num_samples (int, optional): Number of samples to select for splitting. If None, use all files.
    """
    
    if not is_empty(train_path) or not is_empty(test_path) or not is_empty(val_path):
        print("One or more target directories are not empty. Skipping dataset split.")
        return

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    if iam:
        all_files = [f for f in src_path.glob('**/*') if f.is_file()]
    else:
        all_files = [f for f in src_path.iterdir() if f.is_file()]
    
    print("File names loaded…")
    
    if num_samples is not None and len(all_files) < num_samples:
        print(f"Not enough files in the source directory. Found {len(all_files)} files.")
        return
    
    if num_samples is not None:
        random.shuffle(all_files)
        selected_files = all_files[:num_samples]
    else:
        selected_files = all_files

    total_files = len(selected_files)
    train_end = int(total_files * train_ratio)
    test_end = train_end + int(total_files * test_ratio)
    
    train_files = selected_files[:train_end]
    test_files = selected_files[train_end:test_end]
    val_files = selected_files[test_end:]

    print("Start copying files…")

    def copy_files(files, target_path, set_name):
        for i, file in enumerate(files, 1):
            target_file_path = target_path / file.name
            shutil.copy(file, target_file_path)
            if i % 100 == 0 or i == len(files):
                print(f"Copied {i}/{len(files)} files to {set_name}")

    copy_files(train_files, train_path, "train")
    copy_files(test_files, test_path, "test")
    copy_files(val_files, val_path, "val")

    print(f"Dataset split completed: {len(train_files)} train, {len(test_files)} test, {len(val_files)} val files.")


class CaptchaLoader(Dataset):
    def __init__(self, img_dir: str):
        """
        Initialize the dataset with image directory and transformations.

        Args:
            img_dir (str): Path to the directory containing captcha images.
        """
        self.img_dir = os.path.abspath(img_dir)
        self.paths = [os.path.join(self.img_dir, filename) for filename in os.listdir(self.img_dir)]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.
        """
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Fetch an image and its corresponding text label.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            Tuple[torch.Tensor, str]: A tuple containing the transformed image and the associated text.
        """
        path = self.paths[idx]
        text = self.get_filename(path)
        try:
            img = Image.open(path).convert('RGB')
        except IOError:
            raise ValueError(f"Error opening image file {path}")
        
        img = self.transform(img)
        return img, text
    
    def get_filename(self, path: str) -> str:
        """
        Extracts the filename (without extension) and converts it to lowercase.

        Args:
            path (str): Full path of the image file.

        Returns:
            str: Filename without extension, in lowercase.
        """
        return os.path.splitext(os.path.basename(path))[0].lower().strip()


class IAMLoader(Dataset):
    """
    A custom dataset class for loading and processing the IAM Handwriting Database.
    """

    def __init__(self, img_dir: str, information_file: str):
        """
        Initializes the IAMLoader with the specified image directory and information file.

        Args:
            img_dir (str): The directory containing the image files.
            information_file (str): The file containing transcriptions and other information.
        """
        self.img_dir = os.path.abspath(img_dir)
        self.paths = [os.path.join(self.img_dir, filename) for filename in os.listdir(self.img_dir) if filename.endswith(('.png', '.jpg', '.jpeg'))]
        self.information_file = os.path.abspath(information_file)
        self.transform = transforms.Compose([
            transforms.Lambda(self.resize_and_pad),  # Custom transformation
            transforms.ToTensor()
        ])
        self.transcriptions = self.parse_information_file()

    def parse_information_file(self) -> dict:
        """
        Parses the information file to extract transcriptions.

        Returns:
            dict: A dictionary mapping image filenames to their transcriptions.
        """
        transcriptions = {}
        with open(self.information_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    word_id = parts[0]
                    transcription = parts[-1]
                    transcriptions[word_id] = transcription
        return transcriptions

    def resize_and_pad(self, image: Image.Image) -> Image.Image:
        """
        Resizes and pads the image to fit within the target dimensions while maintaining the aspect ratio.

        Args:
            image (Image.Image): The input image to be resized and padded.

        Returns:
            Image.Image: The resized and padded image.
        """
        target_width, target_height = 200, 50
        original_width, original_height = image.size
        
        # Calculate the scaling factor
        scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate the new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the image
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Create a new image with the target size and paste the resized image onto it
        new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        new_image.paste(image, (0, (target_height - new_height) // 2))
        
        return new_image

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.paths)

    def get_filename(self, path: str) -> str:
        """
        Extracts the filename without the extension from a given path.

        Args:
            path (str): The full path to the file.

        Returns:
            str: The filename without the extension.
        """
        return os.path.splitext(os.path.basename(path))[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieves the image and its corresponding transcription at the specified index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, str]: The transformed image and its transcription.
        """
        path = self.paths[idx]
        filename = self.get_filename(path)
        transcription = self.transcriptions.get(filename, "")

        try:
            img = Image.open(path).convert('RGB')
        except IOError:
            raise ValueError(f"Error opening image file {path}")

        img = self.transform(img)
        return img, transcription

    def extract_unique_chars(self) -> set:
        """
        Extracts all unique characters from the transcriptions.

        Returns:
            set: A set of unique characters found in the transcriptions.
        """
        unique_chars = set()

        for transcription in self.transcriptions.values():
            unique_chars.update(c for c in transcription)

        return unique_chars
