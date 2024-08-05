import torch
import numpy as np

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.columns import Columns
from rich.panel import Panel
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

from plotting import print_sample_grid
from utils import decode_logits

class CTCModelTesterEvaluator:
    """
    A class for evaluating and testing a CTC model.
    """

    def __init__(self, model, test_dataset, label_encoder, device, alphabet) -> None:
        """
        Initializes the CTCModelTesterEvaluator with model, dataset, and other parameters.

        Args:
            model: The CTC model to be tested.
            test_dataset: Dataset to be used for testing the model.
            label_encoder: Encoder used for decoding model predictions.
            device: Device on which the model and data are loaded (CPU or GPU).
            alphabet: List of characters representing the alphabet used in the dataset.
        """
        self.model = model
        self.test_dataset = test_dataset
        self.label_encoder = label_encoder
        self.device = device
        self.alphabet = alphabet

        self.test_results = []  # [(Gold, Prediction), ...]

        self.console = Console()


    def test(self):
        """
        Tests the CTC model on the provided test dataset.

        This method evaluates the model and stores the results (gold text and predicted text)
        in the `test_results` attribute.
        """
        self.test_results = []

        progress = Progress(
            SpinnerColumn(spinner_name="earth"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="black"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
            console=self.console,
        )
        
        with progress:
            with torch.no_grad():
                task = progress.add_task(f"Testing", total=(len(self.test_dataset)-1))
                self.model.eval()
                for idx in range(len(self.test_dataset)):
                    img, text = self.test_dataset[idx]
                    logits = self.model(img.unsqueeze(0).to(self.device))
                    pred_text = decode_logits(logits.cpu(), self.label_encoder)

                    self.test_results.append((text, pred_text))

                    progress.update(task, advance=1)


    def word_accuracy(self):
        """
        Calculates the word-level accuracy of the model on the test dataset.

        Returns:
            float: The word-level accuracy as a proportion of correct predictions.
        
        Raises:
            RuntimeError: If `test()` method has not been run before calling this method.
        """
        if self.test_results is None:
            raise RuntimeError("Test method needs to be run before calculating accuracy!")
        
        hits = 0
        for gold, prediction in self.test_results:
            if gold == prediction:
                hits += 1
        return hits / len(self.test_dataset)


    def char_level_bleu(self):
        """
        Calculates the character-level BLEU score of the model on the test dataset.

        Returns:
            float: The average BLEU score across all test samples.
        
        Raises:
            RuntimeError: If `test()` method has not been run before calling this method.
        """
        if self.test_results is None:
            raise RuntimeError("Test method needs to be run before calculating BLEU score!")
        
        total_bleu_score = 0
        for gold, prediction in self.test_results:
            reference = list(gold)
            hypothesis = list(prediction)
            bleu_score = sentence_bleu([reference], hypothesis, weights=(1,))
            total_bleu_score += bleu_score
        
        return total_bleu_score / len(self.test_results)
    

    def print_confusion_matrix(self):
        """
        Prints the confusion matrix of character predictions.

        Displays a confusion matrix representing the model's performance on character predictions.
        
        Raises:
            RuntimeError: If `test()` method has not been run before calling this method.
        """
        if self.test_results is None:
            raise RuntimeError("Test method needs to be run before calculating confusion matrix!")

        char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}

        matrix_size = len(self.alphabet)
        confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        for gold, prediction in self.test_results:
            for g_char, p_char in zip(gold, prediction):
                if g_char in char_to_idx and p_char in char_to_idx:
                    g_idx = char_to_idx[g_char]
                    p_idx = char_to_idx[p_char]
                    confusion_matrix[g_idx][p_idx] += 1

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(confusion_matrix, cmap='plasma')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(self.alphabet)))
        ax.set_yticks(np.arange(len(self.alphabet)))
        ax.set_xticklabels(self.alphabet)
        ax.set_yticklabels(self.alphabet)
        
        plt.xticks(rotation=90)
        plt.show()


    def print_results(self):
        """
        Prints evaluation results including word-level accuracy and character-level BLEU score.

        This method also prints a sample grid of model predictions alongside the test dataset.
        
        Raises:
            RuntimeError: If `test()` method has not been run before calling this method.
        """
        if self.test_results is None:
            raise RuntimeError("Test method needs to be run before printing results!")
        
        print_sample_grid(self.model, self.test_dataset, self.label_encoder, self.device)

        results = [
            Panel(f"[bold cyan]Word-level Accuracy:[/bold cyan] {self.word_accuracy():.4f}", expand=True),
            Panel(f"[bold magenta]Char-level BLEU Score:[/bold magenta] {self.char_level_bleu():.4f}", expand=True),
        ]
        self.console.print(Columns(results))
