"""RNN models for body-state estimation via path integration.

Tasks:
1. Define RNN architecture (vanilla RNN, LSTM, or GRU)
2. Implement place cell output layer
3. Define training objective (cross-entropy or other)
4. Provide method to extract embeddings for Team RL
"""

from typing import Any, Literal, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """Vanilla RNN for state estimation.

    Takes angular velocities as input and outputs place cell activations.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        activation: Literal["tanh", "relu"] = "tanh",
        dropout: float = 0.0,
    ):
        """Initialize the RNN.

        Args:
            input_size: Dimension of input (angular velocities).
            hidden_size: Number of hidden units.
            output_size: Number of place cells (output dimension).
            activation: Activation function.
            dropout: Dropout rate applied to RNN output before projection.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity=activation,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            hidden: Optional initial hidden state.

        Returns:
            Tuple of (place_cell_logits, hidden_states).
            place_cell_logits: Shape (batch, seq_len, output_size).
            hidden_states: Shape (batch, seq_len, hidden_size).
        """
        rnn_out, _ = self.rnn(x, hidden)
        normed = self.layer_norm(rnn_out)
        output = self.output_proj(self.dropout(normed))
        return output, rnn_out


class LSTM(nn.Module):
    """LSTM for state estimation."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            hidden: Optional initial hidden state.

        Returns:
            Tuple of (place_cell_logits, hidden_states).
        """
        rnn_out, _ = self.rnn(x, hidden)
        normed = self.layer_norm(rnn_out)
        output = self.output_proj(self.output_dropout(normed))
        return output, rnn_out


class GRU(nn.Module):
    """GRU for state estimation."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            hidden: Optional initial hidden state.

        Returns:
            Tuple of (place_cell_logits, hidden_states).
        """
        rnn_out, _ = self.rnn(x, hidden)
        normed = self.layer_norm(rnn_out)
        output = self.output_proj(self.output_dropout(normed))
        return output, rnn_out


class StateEstimationModel(L.LightningModule):
    """Lightning wrapper for state estimation models.

    Handles training, validation, and provides interface methods
    for other teams (get_embedding, get_hidden_states).
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        model_type: Literal["rnn", "lstm", "gru"] = "rnn",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        use_init_pos: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize the Lightning module.

        Args:
            input_size: Dimension of input (angular velocities).
            hidden_size: Number of hidden units.
            output_size: Number of place cells.
            model_type: Which architecture to use.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            use_init_pos: Whether to encode initial position into h0.
            dropout: Dropout rate on RNN output (0.5 recommended per Banino 2018).
        """
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_init_pos = use_init_pos
        self.model_type = model_type

        # Instantiate the appropriate model
        self.model: Union[RNN, LSTM, GRU]
        if model_type == "rnn":
            self.model = RNN(input_size, hidden_size, output_size, dropout=dropout)
        elif model_type == "lstm":
            self.model = LSTM(input_size, hidden_size, output_size, dropout=dropout)
        elif model_type == "gru":
            self.model = GRU(input_size, hidden_size, output_size, dropout=dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Number of cells per joint (output is two concatenated distributions)
        assert output_size % 2 == 0, "output_size must be even (split between 2 joints)"
        self.cells_per_joint = output_size // 2

        # Encoder for initial position → initial hidden state
        if use_init_pos:
            self.init_encoder = nn.Linear(output_size, hidden_size)

        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

        # Track best metrics for WandB summary
        self._best_val_loss = float("inf")
        self._best_val_acc = 0.0

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x, hidden)

    def _encode_init_pos(self, init_pc: torch.Tensor) -> Any:
        """Encode initial place cell activations into initial hidden state.

        Args:
            init_pc: Initial place cell activations of shape (batch, output_size).

        Returns:
            Initial hidden state of shape (1, batch, hidden_size) for RNN/GRU,
            or tuple of (h0, c0) each (1, batch, hidden_size) for LSTM.
        """
        # (batch, output_size) → (batch, hidden_size)
        h0 = self.init_encoder(init_pc)
        # RNN expects (num_layers, batch, hidden_size)
        h0 = h0.unsqueeze(0)

        if self.model_type == "lstm":
            # LSTM needs both hidden state and cell state
            c0 = torch.zeros_like(h0)
            return (h0, c0)
        return h0

    def _shared_step(
        self, batch: tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward logic for training and validation.

        Targets are two concatenated distributions (one per joint), so we
        compute separate KL divergence for each joint's softmax and average.
        """
        if self.use_init_pos:
            velocities, targets, init_pc = batch
            h0 = self._encode_init_pos(init_pc)
        else:
            velocities, targets = batch
            h0 = None

        logits, _ = self(velocities, hidden=h0)

        logits_flat = logits.view(-1, self.output_size)
        targets_flat = targets.view(-1, self.output_size)

        # Split into per-joint halves
        cpj = self.cells_per_joint
        logits_j1, logits_j2 = logits_flat[:, :cpj], logits_flat[:, cpj:]
        targets_j1, targets_j2 = targets_flat[:, :cpj], targets_flat[:, cpj:]

        # Separate softmax + KL divergence per joint
        log_probs_j1 = F.log_softmax(logits_j1, dim=-1)
        log_probs_j2 = F.log_softmax(logits_j2, dim=-1)
        loss = (
            self.loss_fn(log_probs_j1, targets_j1)
            + self.loss_fn(log_probs_j2, targets_j2)
        ) / 2

        # Top-1 accuracy per joint (averaged)
        acc_j1 = (logits_j1.argmax(dim=-1) == targets_j1.argmax(dim=-1)).float().mean()
        acc_j2 = (logits_j2.argmax(dim=-1) == targets_j2.argmax(dim=-1)).float().mean()
        acc = (acc_j1 + acc_j2) / 2

        return loss, acc, logits

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute a training step."""
        loss, acc, _ = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute a validation step."""
        loss, acc, _ = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log best val metrics for easy WandB summary."""
        val_loss = self.trainer.callback_metrics.get("val/loss")
        val_acc = self.trainer.callback_metrics.get("val/acc")
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = float(val_loss)
        if val_acc is not None and val_acc > self._best_val_acc:
            self._best_val_acc = float(val_acc)
        self.log("val/best_loss", self._best_val_loss, prog_bar=False)
        self.log("val/best_acc", self._best_val_acc, prog_bar=False)

    def configure_optimizers(self):
        """Configure optimizer and cosine annealing scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # =========================================================================
    # Interface methods for other teams
    # =========================================================================

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for RL (Team RL interface).

        Returns the final hidden state as the embedding.

        Args:
            x: Input trajectory of shape (batch, seq_len, input_size).

        Returns:
            Embedding of shape (batch, hidden_size).
        """
        _, hidden_states = self(x)
        return hidden_states[:, -1, :]

    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Get full hidden state trajectory (Team Interpretation interface).

        Args:
            x: Input trajectory of shape (batch, seq_len, input_size).

        Returns:
            Hidden states of shape (batch, seq_len, hidden_size).
        """
        _, hidden_states = self(x)
        return hidden_states

    @classmethod
    def load_for_embedding(cls, checkpoint_path: str) -> "StateEstimationModel":
        """Load a trained model for embedding extraction.

        Args:
            checkpoint_path: Path to model checkpoint.

        Returns:
            Loaded model in eval mode.
        """
        model = cls.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model
