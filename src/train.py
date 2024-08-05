import torch

import numpy as np
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.columns import Columns
from rich.panel import Panel

from plotting import print_sample_image, plot_loss

class CTCModelTrainer:
    """Handles the training and validation of a CTC model."""

    def __init__(self, model, optimizer, criterion, label_encoder, train_dataloader, val_dataloader, model_path, scheduler=None, clip_norm=5.0, batch_size=32):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.label_encoder = label_encoder
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_path = model_path 
        self.scheduler = scheduler
        self.clip_norm = clip_norm
        self.batch_size = batch_size
        self.device = next(model.parameters()).device 

        self.validation_losses = []
        self.train_losses = []
        self.current_epoch = 0
        self.num_epochs = 0
        self.console = Console()

    def _compute_ctc_loss(self, logits: torch.Tensor, texts: list) -> torch.Tensor:
        """Calculates the CTC loss for given logits and texts."""
        input_len = logits.size(0)
        logits = logits.log_softmax(dim=2)
        encoded_texts, text_lengths = self.label_encoder.encode_texts(texts)
        logits_lengths = torch.full(size=(logits.size(1),), fill_value=input_len, dtype=torch.int32, device=self.device)
        return self.criterion(logits, encoded_texts, logits_lengths, text_lengths.to(self.device))

    def _validate_model(self) -> list:
        """Validates the model on the validation dataset."""
        validation_losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_img, batch_text in self.val_dataloader:
                logits = self.model(batch_img.to(self.device))
                loss = self._compute_ctc_loss(logits, batch_text)
                validation_losses.append(loss.item())

        return validation_losses

    def _train_epoch(self) -> tuple:
        """Trains the model for one epoch and returns the average training loss and losses."""
        self.model.train()
        train_loss_epoch = 0
        train_losses = []

        progress = Progress(
            SpinnerColumn(spinner_name="monkey"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="black"),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("[bold magenta]Batch Loss: {task.fields[train_loss]:.4f}[/bold magenta]"),
            TimeRemainingColumn(),
            console=self.console,
        )

        with progress:
            task = progress.add_task(f"Training Epoch {self.current_epoch+1}/{self.num_epochs}", total=len(self.train_dataloader), train_loss=0)

            for batch_imgs, batch_text in self.train_dataloader:
                self.optimizer.zero_grad()
                logits = self.model(batch_imgs.to(self.device))

                # Calculate loss
                train_loss = self._compute_ctc_loss(logits, batch_text)

                if np.isnan(train_loss.detach().cpu().numpy()):
                    continue

                train_losses.append(train_loss.item())
                train_loss_epoch += train_loss.item()

                # Backward pass
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

                progress.update(task, advance=1, train_loss=train_loss.item())

        return train_loss_epoch / len(self.train_dataloader), train_losses

    def _get_current_lr(self):
        """Gets the current learning rate from the optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, num_epochs: int):
        """Trains the model for a given number of epochs."""
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.train_losses = []
        self.validation_losses = []

        try:
            while self.current_epoch < self.num_epochs:
                # Train for one epoch
                train_loss, epoch_train_losses = self._train_epoch()
                self.train_losses.extend(epoch_train_losses)

                # Validate the model
                val_losses = self._validate_model()
                self.validation_losses.extend(val_losses)
                val_loss = sum(val_losses) / len(val_losses) if val_losses else 0

                # Print progress
                plot_loss(self.current_epoch+1, self.train_losses, self.validation_losses)
                print_sample_image(self.model, self.val_dataloader.dataset, self.device, self.label_encoder)

                # Print the epoch results using rich in columns
                current_lr = self._get_current_lr()
                epoch_info = [
                    Panel(f"[bold cyan]Epoch:[/bold cyan] {self.current_epoch+1}/{self.num_epochs}", expand=True),
                    Panel(f"[bold magenta]Train Loss:[/bold magenta] {train_loss:.4f}", expand=True),
                    Panel(f"[bold yellow]Val Loss:[/bold yellow] {val_loss:.4f}", expand=True),
                    Panel(f"[bold green]Learning Rate:[/bold green] {current_lr:.8f}", expand=True)
                ]
                self.console.print(Columns(epoch_info))

                # Step the scheduler if it exists
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Save the model after each epoch
                self.save_model()

                self.current_epoch += 1

        except KeyboardInterrupt:
            print("Training interrupted.")
            self.save_model()

    def save_model(self):
        """Saves the model's state dictionary to the specified path."""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
