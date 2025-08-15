import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class MultiTokenPredictionDataset(Dataset):
    """
    Dataset class for Multi-Token Prediction (MTP) training as described in
    "Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential"

    This dataset modifies input sequences by inserting k mask tokens after NTP tokens
    at different positions to enable parallel training of multiple prediction heads.
    """

    def __init__(
        self,
        sequences: List[List[int]],  # List of tokenized sequences
        tokenizer: AutoTokenizer,
        num_masks: int = 8,
        max_length: int = 2048,
        mask_token_ids: Optional[List[int]] = None,
        ignore_index: int = -100,
        mask_probability: float = 1.0  # Probability of inserting masks after each token
    ):
        """
        Initialize the MTP Dataset

        Args:
            sequences: List of tokenized sequences (list of token ids)
            tokenizer: Tokenizer used for the model
            num_masks: Number of mask tokens to insert (k in the paper)
            max_length: Maximum sequence length
            mask_token_ids: List of mask token IDs. If None, will create new ones
            ignore_index: Index to ignore in loss calculation
            mask_probability: Probability of inserting masks after each NTP token
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.num_masks = num_masks
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.mask_probability = mask_probability

        # Create or use provided mask token IDs
        if mask_token_ids is None:
            # Create unique mask tokens (m1, m2, ..., mk)
            vocab_size = len(tokenizer)
            self.mask_token_ids = list(range(vocab_size, vocab_size + num_masks))
        else:
            assert len(mask_token_ids) == num_masks, "Number of mask tokens must match num_masks"
            self.mask_token_ids = mask_token_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with mask tokens inserted

        Returns:
            Dictionary containing:
            - input_ids: Modified input sequence with mask tokens
            - labels: Target labels for both NTP and MTP tokens
            - attention_mask: Attention mask for the sequence
            - position_ids: Position IDs for the sequence
            - ntp_mask: Binary mask indicating NTP token positions
            - mtp_mask: Binary mask indicating MTP token positions
        """
        original_sequence = self.sequences[idx]

        # Truncate if too long (reserve space for potential mask insertions)
        max_original_length = self.max_length // (self.num_masks + 1)
        if len(original_sequence) > max_original_length:
            original_sequence = original_sequence[:max_original_length]

        # Create modified input with mask tokens
        modified_input, labels, position_ids, ntp_mask, mtp_mask = self._create_masked_input(
            original_sequence
        )

        # Pad sequences to max_length
        modified_input, labels, position_ids, ntp_mask, mtp_mask, attention_bias = self._pad_sequences(
            modified_input, labels, position_ids, ntp_mask, mtp_mask
        )

        return {
            'input_ids': torch.tensor(modified_input, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_bias': attention_bias,  # 2D matrix [seq_len, seq_len]
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'ntp_mask': torch.tensor(ntp_mask, dtype=torch.bool),
            'mtp_mask': torch.tensor(mtp_mask, dtype=torch.bool)
        }

    def _create_masked_input(self, sequence: List[int]) -> Tuple[List[int], List[int], List[int], List[bool], List[bool]]:
        """
        Create masked input following the paper's methodology.
        For sequence of length n, creates n separate prompts where i-th prompt
        includes first i tokens followed by k masks.
        """
        modified_input = []
        labels = []
        position_ids = []
        ntp_mask = []  # True for NTP tokens
        mtp_mask = []  # True for MTP tokens

        for i in range(len(sequence)):
            # Add the i-th NTP token
            modified_input.append(sequence[i])
            labels.append(sequence[i] if i > 0 else self.ignore_index)  # First token usually ignored
            position_ids.append(i)
            ntp_mask.append(True)
            mtp_mask.append(False)

            # Insert k mask tokens after this position (with probability)
            if i < len(sequence) - 1 and np.random.random() < self.mask_probability:
                for j in range(self.num_masks):
                    # Add mask token
                    modified_input.append(self.mask_token_ids[j])

                    # Label is the future token at position i+1+j
                    future_pos = i + 1 + j
                    if future_pos < len(sequence):
                        labels.append(sequence[future_pos])
                    else:
                        labels.append(self.ignore_index)

                    # Position ID continues from current position
                    position_ids.append(i + 1 + j)
                    ntp_mask.append(False)
                    mtp_mask.append(True)

        return modified_input, labels, position_ids, ntp_mask, mtp_mask

    def _create_attention_bias_matrix(
        self,
        input_ids: List[int],
        ntp_mask: List[bool],
        mtp_mask: List[bool],
        seq_len: int,
        actual_length: int
    ) -> torch.Tensor:
        """
        Create 2D attention bias matrix following the paper's attention pattern:
        - NTP tokens attend only to previous NTP tokens
        - MTP tokens attend to previous NTP tokens and MTP tokens of the same block,
          but not to earlier MTP blocks
        - Padded positions are masked out

        Returns:
            attention_bias: [seq_len, seq_len] matrix where 1 = can attend, 0 = cannot attend
        """
        attention_bias = torch.zeros((seq_len, seq_len), dtype=torch.float)

        # First, handle padding mask - padded positions cannot attend or be attended to
        for i in range(actual_length):
            for j in range(actual_length):
                attention_bias[i, j] = 1.0  # Initialize all actual positions as attendable

        # Now apply the MTP attention rules
        ntp_positions = [i for i, is_ntp in enumerate(ntp_mask[:actual_length]) if is_ntp]
        mtp_positions = [i for i, is_mtp in enumerate(mtp_mask[:actual_length]) if is_mtp]

        # Group MTP tokens into blocks (tokens inserted after the same NTP position)
        mtp_blocks = self._group_mtp_tokens(ntp_positions, mtp_positions, actual_length)

        # Reset attention matrix and apply rules
        attention_bias.fill_(0.0)

        # Rule 1: NTP tokens can attend to all previous NTP tokens
        for i, pos_i in enumerate(ntp_positions):
            for j, pos_j in enumerate(ntp_positions):
                if pos_j <= pos_i:  # Can attend to current and previous NTP tokens
                    attention_bias[pos_i, pos_j] = 1.0

        # Rule 2: MTP tokens attention pattern
        for block_idx, (block_start_ntp, mtp_block) in enumerate(mtp_blocks):
            for i, mtp_pos in enumerate(mtp_block):
                # Can attend to all NTP tokens up to the NTP token this block follows
                for ntp_pos in ntp_positions:
                    if ntp_pos <= block_start_ntp:
                        attention_bias[mtp_pos, ntp_pos] = 1.0

                # Can attend to previous MTP tokens in the same block
                for j in range(i + 1):  # Including self-attention
                    attention_bias[mtp_pos, mtp_block[j]] = 1.0

        # Rule 3: All tokens can attend to themselves (if not already covered)
        for i in range(actual_length):
            attention_bias[i, i] = 1.0

        return attention_bias

    def _group_mtp_tokens(
        self,
        ntp_positions: List[int],
        mtp_positions: List[int],
        seq_len: int
    ) -> List[Tuple[int, List[int]]]:
        """
        Group MTP tokens into blocks based on which NTP token they follow

        Returns:
            List of (ntp_position, mtp_block) tuples
        """
        mtp_blocks = []

        for i, ntp_pos in enumerate(ntp_positions[:-1]):  # Exclude last NTP as it has no following MTP
            next_ntp_pos = ntp_positions[i + 1] if i + 1 < len(ntp_positions) else seq_len

            # Find MTP tokens between this NTP and the next NTP
            block_mtp = [pos for pos in mtp_positions if ntp_pos < pos < next_ntp_pos]

            if block_mtp:
                mtp_blocks.append((ntp_pos, block_mtp))

        return mtp_blocks

    def _pad_sequences(
        self,
        input_ids: List[int],
        labels: List[int],
        position_ids: List[int],
        ntp_mask: List[bool],
        mtp_mask: List[bool]
    ) -> Tuple[List[int], List[int], List[int], List[bool], List[bool], torch.Tensor]:
        """
        Pad sequences to max_length and create proper 2D attention bias
        """
        current_length = len(input_ids)

        if current_length > self.max_length:
            # Truncate if too long
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            position_ids = position_ids[:self.max_length]
            ntp_mask = ntp_mask[:self.max_length]
            mtp_mask = mtp_mask[:self.max_length]
            final_length = self.max_length
        else:
            # Pad if too short
            pad_length = self.max_length - current_length

            input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
            labels.extend([self.ignore_index] * pad_length)
            position_ids.extend([0] * pad_length)  # Pad with 0s
            ntp_mask.extend([False] * pad_length)
            mtp_mask.extend([False] * pad_length)
            final_length = self.max_length

        # Create 2D attention bias matrix
        attention_bias = self._create_attention_bias_matrix(
            input_ids, ntp_mask, mtp_mask, final_length, current_length
        )

        return input_ids, labels, position_ids, ntp_mask, mtp_mask, attention_bias

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader
        """
        # Stack all tensors
        collated = {}
        for key in batch[0].keys():
            collated[key] = torch.stack([item[key] for item in batch])

        return collated

    def get_mask_token_ids(self) -> List[int]:
        """Return the mask token IDs used by this dataset"""
        return self.mask_token_ids

    def create_inference_input(self, input_ids: List[int], num_masks: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Create input for inference with mask tokens appended at the end

        Args:
            input_ids: Input token sequence
            num_masks: Number of masks to append (defaults to self.num_masks)

        Returns:
            Dictionary with input_ids, attention_mask, and position_ids for inference
        """
        if num_masks is None:
            num_masks = self.num_masks

        # Append mask tokens at the end
        inference_input = input_ids + self.mask_token_ids[:num_masks]

        # Create attention mask (all 1s for inference)
        attention_mask = [1] * len(inference_input)

        # Create position IDs
        position_ids = list(range(len(inference_input)))

        # Create masks for NTP/MTP distinction
        ntp_mask = [True] * len(input_ids) + [False] * num_masks
        mtp_mask = [False] * len(input_ids) + [True] * num_masks

        return {
            'input_ids': torch.tensor([inference_input], dtype=torch.long),
            'attention_bias': torch.ones((1, len(inference_input), len(inference_input)), dtype=torch.float),  # Full attention for inference
            'position_ids': torch.tensor([position_ids], dtype=torch.long),
            'ntp_mask': torch.tensor([ntp_mask], dtype=torch.bool),
            'mtp_mask': torch.tensor([mtp_mask], dtype=torch.bool)
        }


# Example usage and helper functions
def create_mtp_dataset_from_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    num_masks: int = 8,
    max_length: int = 2048
) -> MultiTokenPredictionDataset:
    """
    Create MTP dataset from raw texts

    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        num_masks: Number of mask tokens
        max_length: Maximum sequence length

    Returns:
        MultiTokenPredictionDataset instance
    """
    # Tokenize texts
    sequences = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        sequences.append(tokens)

    return MultiTokenPredictionDataset(
        sequences=sequences,
        tokenizer=tokenizer,
        num_masks=num_masks,
        max_length=max_length
    )
