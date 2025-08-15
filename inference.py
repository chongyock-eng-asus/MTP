import torch
from typing import Dict, List, Optional

def mtp_inference(
    model,
    tokenizer,
    input_text: str,
    mask_token_ids: List[int],
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> Dict:
    """
    Simple MTP inference with linear decoding

    Args:
        model: Your trained MTP model
        tokenizer: Tokenizer
        input_text: Input text to continue
        mask_token_ids: List of mask token IDs (e.g., [50001, 50002, ...])
        max_new_tokens: Maximum tokens to generate
        device: Device to run on

    Returns:
        Dictionary with generated text and metrics
    """

    model.eval()

    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    current_sequence = input_ids.clone()
    original_length = input_ids.size(1)

    total_steps = 0
    total_accepted = 0

    with torch.no_grad():
        while current_sequence.size(1) - original_length < max_new_tokens:
            total_steps += 1

            # Step 1: Append mask tokens to current sequence
            input_with_masks = torch.cat([
                current_sequence,
                torch.tensor([mask_token_ids], device=device)
            ], dim=1)

            # Step 2: Forward pass through model
            outputs = model(input_ids=input_with_masks)

            # Step 3: Get predictions for k+1 positions
            start_pos = current_sequence.size(1)

            # Handle different output formats
            if isinstance(outputs, dict):
                # If model returns dict (like our MTP model)
                if 'base_logits' in outputs:
                    logits = outputs['base_logits']
                elif 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    # Try to find logits in the dict
                    logits = list(outputs.values())[0]  # First tensor in dict
            else:
                # If model returns object with .logits attribute
                logits = outputs.logits

            predicted_logits = logits[0, start_pos:start_pos + len(mask_token_ids) + 1]
            predicted_tokens = torch.argmax(predicted_logits, dim=-1)

            # Step 4: Linear verification
            verified_tokens = []

            # First token always accepted (autoregressive)
            verified_tokens.append(predicted_tokens[0].item())

            # Verify remaining tokens
            for i in range(1, len(predicted_tokens)):
                # Create test sequence
                test_sequence = torch.cat([
                    current_sequence,
                    torch.tensor([verified_tokens], device=device)
                ], dim=1)

                # Get autoregressive prediction
                test_outputs = model(input_ids=test_sequence)

                # Handle different output formats
                if isinstance(test_outputs, dict):
                    if 'base_logits' in test_outputs:
                        test_logits = test_outputs['base_logits']
                    elif 'logits' in test_outputs:
                        test_logits = test_outputs['logits']
                    else:
                        test_logits = list(test_outputs.values())[0]
                else:
                    test_logits = test_outputs.logits

                autoregressive_token = torch.argmax(test_logits[0, -1, :], dim=-1)

                # Check if matches
                if autoregressive_token == predicted_tokens[i]:
                    verified_tokens.append(predicted_tokens[i].item())
                else:
                    break  # Stop at first mismatch

            # Add verified tokens to sequence
            current_sequence = torch.cat([
                current_sequence,
                torch.tensor([verified_tokens], device=device)
            ], dim=1)

            total_accepted += len(verified_tokens)

            # Check for EOS
            if tokenizer.eos_token_id and verified_tokens[-1] == tokenizer.eos_token_id:
                break

    # Results
    generated_text = tokenizer.decode(current_sequence[0])
    total_generated = current_sequence.size(1) - original_length
    acceptance_rate = total_accepted / total_steps if total_steps > 0 else 0
    speedup = total_generated / total_steps if total_steps > 0 else 1

    return {
        'input_text': input_text,
        'generated_text': generated_text,
        'new_tokens_only': tokenizer.decode(current_sequence[0][original_length:]),
        'total_steps': total_steps,
        'total_generated': total_generated,
        'acceptance_rate': acceptance_rate,
        'speedup': speedup
    }