import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from typing import Dict, List, Optional


def train_mtp_model(
    model,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_steps: int = 1000,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    save_steps: int = 1000,
    eval_steps: int = 500,
    logging_steps: int = 100,
    output_dir: str = "./mtp_checkpoints",
    device: str = "cuda"
):
    """
    Simple training loop for Multi-Token Prediction model

    Args:
        model: MultiTokenPredictionModel instance
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        output_dir: Directory to save checkpoints
        device: Device to train on
    """

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Move model to device
    model.to(device)
    model.train()

    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    logger.info(f"Training {sum(p.numel() for p in trainable_params):,} parameters")

    # Setup optimizer
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Calculate total steps
    total_steps = len(train_dataloader) * num_epochs

    # Setup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training state
    global_step = 0
    total_loss = 0.0
    best_val_loss = float('inf')

    logger.info("Starting training...")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(epoch_iterator):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                ntp_mask=batch.get('ntp_mask'),
                mtp_mask=batch.get('mtp_mask'),
                labels=batch.get('labels')
            )

            loss = outputs['loss']

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    current_lr = scheduler.get_last_lr()[0]

                    logger.info(
                        f"Step {global_step}: "
                        f"loss={avg_loss:.4f}, "
                        f"lr={current_lr:.2e}"
                    )

                    # Update progress bar
                    epoch_iterator.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })

                    total_loss = 0.0

                # Evaluation
                if val_dataloader and global_step % eval_steps == 0:
                    val_loss = evaluate_model(model, val_dataloader, device)
                    logger.info(f"Validation loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, scheduler, global_step,
                                      f"{output_dir}/best_model", logger)

                    model.train()  # Back to training mode

                # Save checkpoint
                if global_step % save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step,
                                  f"{output_dir}/checkpoint-{global_step}", logger)

    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step,
                  f"{output_dir}/final_model", logger)

    logger.info("Training completed!")


def evaluate_model(model, dataloader: DataLoader, device: str) -> float:
    """
    Evaluate the model on validation data

    Args:
        model: Model to evaluate
        dataloader: Validation data loader
        device: Device to evaluate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                ntp_mask=batch.get('ntp_mask'),
                mtp_mask=batch.get('mtp_mask'),
                labels=batch.get('labels')
            )

            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: str, logger):
    """
    Save model checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        output_dir: Directory to save to
        logger: Logger instance
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step
    }, f"{output_dir}/pytorch_model.bin")

    logger.info(f"Checkpoint saved to {output_dir}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, device: str):
    """
    Load model checkpoint

    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        scheduler: Scheduler to load into
        checkpoint_path: Path to checkpoint
        device: Device to load to

    Returns:
        Step number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['step']
