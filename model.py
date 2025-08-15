import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Dict, List
import math


class GatedLoRALinear(nn.Module):
    """
    Gated LoRA layer that only applies adaptations to MTP tokens
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 128, alpha: int = 256, dropout: float = 0.1):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, base_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, mtp_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Base transformation
        base_output = self.base_layer(x)

        if mtp_mask is None:
            return base_output

        # LoRA transformation
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        # Apply gating: only use LoRA for MTP tokens
        mtp_mask_expanded = mtp_mask.unsqueeze(-1).expand_as(base_output)
        gated_output = torch.where(mtp_mask_expanded, base_output + lora_output, base_output)

        return gated_output


class SamplerHead(nn.Module):
    """
    Lightweight sampler head for coherent sequence generation
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size)
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor, previous_embeddings: torch.Tensor) -> torch.Tensor:
        # Concatenate hidden states with previous token embeddings
        combined = torch.cat([hidden_states, previous_embeddings], dim=-1)

        # Pass through MLP
        features = self.mlp(combined)

        # Project to vocabulary
        logits = self.output_projection(features)

        return logits


class MultiTokenPredictionModel(nn.Module):
    """
    Multi-Token Prediction Model with Gated LoRA and Sampler Head
    """
    def __init__(
        self,
        base_model_name: str,
        num_masks: int = 8,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.1
    ):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.num_masks = num_masks

        # Store original vocab size before adding mask tokens
        self.original_vocab_size = self.config.vocab_size

        # Apply Gated LoRA to transformer layers
        self._apply_gated_lora(lora_rank, lora_alpha, lora_dropout)

        # Create sampler head
        self.sampler_head = SamplerHead(
            self.config.hidden_size,
            self.config.vocab_size + num_masks  # Account for new mask tokens
        )

        # Share embedding weights with sampler head output projection
        self.sampler_head.output_projection.weight = self.base_model.lm_head.weight

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Keep LoRA and sampler parameters trainable
        for name, module in self.named_modules():
            if isinstance(module, GatedLoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

        for param in self.sampler_head.parameters():
            param.requires_grad = True

    def _apply_gated_lora(self, rank: int, alpha: int, dropout: float):
        """Apply Gated LoRA to transformer layers"""
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                # Replace with Gated LoRA
                gated_layer = GatedLoRALinear(module, rank, alpha, dropout)

                # Set the layer in the model
                parent_name = '.'.join(name.split('.')[:-1])
                layer_name = name.split('.')[-1]
                parent_module = self.base_model.get_submodule(parent_name)
                setattr(parent_module, layer_name, gated_layer)

    def extend_vocabulary(self, tokenizer):
        """Extend vocabulary with mask tokens"""
        # Add mask tokens
        mask_tokens = [f"<mask_{i}>" for i in range(self.num_masks)]
        num_added = tokenizer.add_tokens(mask_tokens)

        # Resize embeddings
        self.base_model.resize_token_embeddings(len(tokenizer))

        # Update sampler head output size
        old_weight = self.sampler_head.output_projection.weight
        self.sampler_head.output_projection = nn.Linear(
            self.config.hidden_size,
            len(tokenizer)
        )

        # Copy old weights and initialize new ones
        with torch.no_grad():
            self.sampler_head.output_projection.weight[:old_weight.size(0)] = old_weight
            # Initialize new mask token embeddings
            self.sampler_head.output_projection.weight[old_weight.size(0):] = torch.randn(
                num_added, self.config.hidden_size
            ) * 0.02

        # Share weights again
        self.sampler_head.output_projection.weight = self.base_model.lm_head.weight

        # Get mask token IDs
        mask_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in mask_tokens]
        return mask_token_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ntp_mask: Optional[torch.Tensor] = None,
        mtp_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MTP model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            ntp_mask: Mask for NTP tokens [batch_size, seq_len]
            mtp_mask: Mask for MTP tokens [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]
            return_dict: Whether to return dictionary

        Returns:
            Dictionary containing:
            - base_logits: Logits from base model
            - sampler_logits: Logits from sampler head
            - loss: Combined loss (if labels provided)
            - hidden_states: Last layer hidden states
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings for input tokens
        embeddings = self.base_model.get_input_embeddings()

        # Forward pass through base model with LoRA
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )

        hidden_states = outputs.hidden_states[-1]
        base_logits = outputs.logits

        # Prepare previous token embeddings for sampler
        # Shift input_ids to get previous tokens
        shifted_input_ids = torch.cat([
            torch.zeros_like(input_ids[:, :1]),  # Pad first position
            input_ids[:, :-1]
        ], dim=1)

        previous_embeddings = embeddings(shifted_input_ids)

        # Sampler head forward pass
        sampler_logits = self.sampler_head(hidden_states, previous_embeddings)

        # Compute losses if labels provided
        total_loss = None
        base_loss = None
        sampler_loss = None
        lcm_loss = None

        if labels is not None:
            losses = self.compute_losses(
                base_logits, sampler_logits, hidden_states,
                labels, ntp_mask, mtp_mask
            )
            total_loss = losses['total_loss']
            base_loss = losses['base_loss']
            sampler_loss = losses['sampler_loss']
            lcm_loss = losses['lcm_loss']

        if return_dict:
            return {
                'base_logits': base_logits,
                'sampler_logits': sampler_logits,
                'hidden_states': hidden_states,
                'loss': total_loss,
                'base_loss': base_loss,
                'sampler_loss': sampler_loss,
                'lcm_loss': lcm_loss
            }

        return (base_logits, sampler_logits, hidden_states, total_loss)

    def compute_losses(
        self,
        base_logits: torch.Tensor,
        sampler_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        ntp_mask: Optional[torch.Tensor] = None,
        mtp_mask: Optional[torch.Tensor] = None,
        lcm_weight: float = 1.0,
        sampler_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute all training losses"""

        # Shift for causal language modeling
        shift_logits_base = base_logits[..., :-1, :].contiguous()
        shift_logits_sampler = sampler_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Base cross-entropy loss
        base_loss = F.cross_entropy(
            shift_logits_base.view(-1, shift_logits_base.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Sampler cross-entropy loss
        sampler_loss = F.cross_entropy(
            shift_logits_sampler.view(-1, shift_logits_sampler.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Latent Consistency Matching (LCM) loss
        lcm_loss = self.compute_lcm_loss(hidden_states, ntp_mask, mtp_mask)

        # Total loss
        total_loss = base_loss + sampler_weight * sampler_loss + lcm_weight * lcm_loss

        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'sampler_loss': sampler_loss,
            'lcm_loss': lcm_loss
        }

    def compute_lcm_loss(
        self,
        hidden_states: torch.Tensor,
        ntp_mask: Optional[torch.Tensor] = None,
        mtp_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Latent Consistency Matching loss"""
        if ntp_mask is None or mtp_mask is None:
            return torch.tensor(0.0, device=hidden_states.device)

        batch_size, seq_len, hidden_size = hidden_states.shape
        lcm_losses = []

        for b in range(batch_size):
            ntp_positions = torch.where(ntp_mask[b])[0]
            mtp_positions = torch.where(mtp_mask[b])[0]

            for ntp_pos in ntp_positions:
                ntp_hidden = hidden_states[b, ntp_pos]

                # Find MTP tokens that should predict this NTP position
                corresponding_mtp = []
                for mtp_pos in mtp_positions:
                    if mtp_pos < ntp_pos:
                        corresponding_mtp.append(hidden_states[b, mtp_pos])

                if corresponding_mtp:
                    mtp_stack = torch.stack(corresponding_mtp)
                    # L2 distance (detach NTP to prevent gradient flow)
                    distances = torch.norm(mtp_stack - ntp_hidden.detach(), dim=-1) ** 2
                    lcm_losses.append(distances.mean())

        return torch.stack(lcm_losses).mean() if lcm_losses else torch.tensor(0.0, device=hidden_states.device)

    def generate_with_mtp(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        num_masks: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text using multi-token prediction

        Args:
            input_ids: Input token IDs [1, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            num_masks: Number of mask tokens to use (defaults to self.num_masks)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated token IDs [1, seq_len + new_tokens]
        """
        if num_masks is None:
            num_masks = self.num_masks

        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens // (num_masks + 1)):
                # Create input with mask tokens
                mask_token_ids = list(range(
                    self.original_vocab_size,
                    self.original_vocab_size + num_masks
                ))

                input_with_masks = torch.cat([
                    generated,
                    torch.tensor([mask_token_ids], device=generated.device)
                ], dim=1)

                # Forward pass
                outputs = self.forward(
                    input_ids=input_with_masks,
                    return_dict=True
                )

                # Get logits for the mask positions
                logits = outputs['sampler_logits'][0, -num_masks-1:]  # Last num_masks+1 positions

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Sample tokens
                if do_sample:
                    # Top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        logits = logits.masked_fill(indices_to_remove, float('-inf'))

                    probs = F.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)

                # Add generated tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(0)], dim=1)

                # Check for early stopping (e.g., EOS token)
                if hasattr(self.base_model.config, 'eos_token_id'):
                    if (next_tokens == self.base_model.config.eos_token_id).any():
                        break

        return generated

    def get_trainable_parameters(self):
        """Get all trainable parameters"""
        trainable_params = []

        # LoRA parameters
        for name, module in self.named_modules():
            if isinstance(module, GatedLoRALinear):
                trainable_params.extend([module.lora_A, module.lora_B])

        # Sampler head parameters
        trainable_params.extend(self.sampler_head.parameters())

        return [p for p in trainable_params if p.requires_grad]
