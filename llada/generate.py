# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from transformers.cache_utils import DynamicCache
from torch.cuda import nvtx

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def handle_early_stop(x, block_states, eos_token_id, prompt_length, mask_token_id=None, debug=False):
    """
    Early stop: find first EOS, set all tokens after it to EOS (not mask), mark subsequent blocks as complete.
    Returns: (has_eos, first_eos_abs_pos)
    """
    if eos_token_id is None:
        return False, None
    
    # Only check non-mask tokens in generation region
    gen_region = x[:, prompt_length:]
    eos_mask = (gen_region == eos_token_id)
    
    if not eos_mask.any():
        return False, None
    
    # Find first EOS position (only among decoded, non-mask tokens)
    pos = torch.arange(gen_region.shape[1], device=x.device).unsqueeze(0)
    first_eos_rel = torch.where(eos_mask, pos, gen_region.shape[1]).amin(dim=1)
    first_eos_abs = prompt_length + first_eos_rel[0].item()
    
    if debug:
        print(f"[EarlyStop] Found first EOS at position {first_eos_abs} (relative: {first_eos_rel[0].item()})")
        # Count non-mask tokens before EOS
        if mask_token_id is not None:
            decoded_before_eos = ((x[:, prompt_length:first_eos_abs+1] != mask_token_id).sum().item())
            print(f"[EarlyStop] Decoded tokens before (and including) EOS: {decoded_before_eos}")
    
    # Set all tokens after first EOS to EOS (NOT mask!)
    x[:, first_eos_abs+1:] = eos_token_id
    
    # Update block states based on EOS position
    blocks_marked = 0
    blocks_updated = 0
    for bid in sorted(block_states.keys()):
        if bid > 0:
            start, end = block_states[bid]["start"], block_states[bid]["end"]
            
            if start > first_eos_abs:
                # Block entirely after EOS - mark as complete with no masks
                if block_states[bid]["mask_count"] != 0:
                    block_states[bid]["mask_count"] = 0
                    block_states[bid]["is_complete"] = True
                    blocks_marked += 1
            elif end > first_eos_abs and start <= first_eos_abs:
                # Block CONTAINS EOS - recalculate mask_count for portion before EOS
                if mask_token_id is not None:
                    # Only count masks from block start to EOS position (inclusive)
                    masks_before_eos = (x[:, start:first_eos_abs+1] == mask_token_id).sum().item()
                    old_count = block_states[bid]["mask_count"]
                    if masks_before_eos != old_count:
                        block_states[bid]["mask_count"] = masks_before_eos
                        blocks_updated += 1
                        if masks_before_eos == 0:
                            block_states[bid]["is_complete"] = True
    
    if debug:
        if blocks_marked > 0:
            print(f"[EarlyStop] Marked {blocks_marked} blocks after EOS as complete")
        if blocks_updated > 0:
            print(f"[EarlyStop] Updated {blocks_updated} blocks containing EOS")
    
    return True, first_eos_abs

#_fast
def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens

#_dparallel
def get_num_transfer_tokens_dparallel(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

#_fastdllm
@ torch.no_grad()
def generate_fastdllm(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe

def get_window_range_asymmetric(block_start, block_end, left_radius, right_radius, total_length):
    """与带缓存版本完全相同的窗口计算函数"""
    window_start = block_start - left_radius
    window_end = block_end + right_radius
    window_start = max(0, window_start)
    window_end = min(total_length, window_end)
    return window_start, window_end

def compute_block_length(
    logits,                
    predicted_tokens,      
    prompt,                
    gen_length,       
    generated_length,
    default_block_length,
    delimiter_ids=[198],  # default: newline token (11=comma, 13=period, 198=newline)
    delimiter_threshold=float('inf')
):
    """
    Compute adaptive block length based on delimiter confidence.
    Returns the position of the highest-confidence delimiter if above threshold,
    otherwise returns default_block_length.
    """
    prompt_length = prompt.shape[1]
    block_start = prompt_length + generated_length
    remaining_length = gen_length - generated_length
    
    # Create sampling window (25% of gen_length, capped by remaining)
    window_size = min(int(0.25 * gen_length), remaining_length)
    window_tokens = predicted_tokens[0, block_start:block_start + window_size]
    
    # Create mask for delimiter tokens
    delimiter_mask = torch.zeros_like(window_tokens, dtype=torch.bool)
    for token_id in delimiter_ids:
        delimiter_mask |= (window_tokens == token_id)

    # Fallback to default block length if no delimiter is found
    if not torch.any(delimiter_mask):
        return min(default_block_length, remaining_length)

    # Get positions of delimiters in the sequence
    delimiter_pos = block_start + torch.nonzero(delimiter_mask).squeeze(-1)
    
    # Compute confidence for each delimiter
    delimiter_logits = logits[0, delimiter_pos, predicted_tokens[0, delimiter_pos]]
    log_sum_exp = torch.logsumexp(logits[0, delimiter_pos, :], dim=-1)
    delimiter_confidences = torch.exp(delimiter_logits - log_sum_exp)

    # Find the delimiter with highest confidence
    max_confidence, best_idx = torch.max(delimiter_confidences, dim=0)
    max_confidence = max_confidence.item()
    best_delimiter_pos = delimiter_pos[best_idx].item()

    if max_confidence >= delimiter_threshold:
        block_length = best_delimiter_pos - block_start + 1
    else:
        block_length = min(default_block_length, remaining_length)
    return block_length

def get_transfer_index_entropy(logits, temperature, remasking, mask_index, x, num_transfer_tokens, entropy_threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    
    # Calculate entropy instead of confidence
    p = F.softmax(logits.to(torch.float64), dim=-1)
    
    if remasking == 'low_confidence':
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)  # b, l
    elif remasking == 'random':
        # For random remasking, use random entropy values
        entropy = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    
    # Use entropy instead of confidence (note: higher entropy means lower confidence)
    # So we set entropy to +inf for non-masked positions (to exclude them from selection)
    entropy_for_selection = torch.where(mask_index, entropy, torch.inf)
    
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    
    if entropy_threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(entropy_for_selection.shape[0]):
        # Select tokens with lowest entropy (highest confidence)
        # topk with largest=False gives us the smallest values (lowest entropy)
        _, select_index = torch.topk(entropy_for_selection[j], k=num_transfer_tokens[j], largest=False)
        transfer_index[j, select_index] = True
        
        if entropy_threshold is not None:
            # Only keep tokens with entropy below threshold (high confidence)
            # If entropy > threshold, set transfer_index to False
            for k in range(1, num_transfer_tokens[j]):  # Skip the first one (lowest entropy)
                if entropy[j, select_index[k]] > entropy_threshold:
                    transfer_index[j, select_index[k]] = False
    
    return x0, transfer_index

#_dparallel
@ torch.no_grad()
def generate_dparallel(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_dparallel(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0

            if threshold is not None:
                x0, transfer_index = get_transfer_index_entropy(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dparallel(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)

            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
        # if x[:, prompt.shape[1] + (num_block + 1) * block_length-1] == 126081:
        #     x[:, prompt.shape[1] + (num_block + 1) * block_length:] = 126081
        #     break
    return x, nfe


#_fastdllm
@ torch.no_grad()
def generate_with_prefix_cache_fastdllm(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., halo_factor=0,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1

    return x, nfe

#_haloblock with prefix cache
@torch.no_grad()
def generate_with_prefix_cache_haloblock(model, prompt, steps=128, gen_length=128, block_length=128, halo_factor=0, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        nfe += 1

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        # ========== calculate decoding difficulty ==========
        s, e = current_block_start, current_block_end
        logits_blk_init = output.logits[:, s:e, :]

        pos_idx = torch.arange(block_length, device=x.device)
        left_weight = 1.0 - (pos_idx.float() / (block_length - 1))

        if halo_factor != 0:
            probs_blk = torch.softmax(logits_blk_init / 10, dim=-1)
        else:
            probs_blk = torch.softmax(logits_blk_init, dim=-1)
        entropy_blk = torch.special.entr(probs_blk).sum(dim=-1)
        mask_blk_init = (x[:, s:e] == mask_id)
        token_uncertainty = entropy_blk * halo_factor

        mask_blk_float = mask_blk_init.float()
        masked_uncertainty = token_uncertainty * mask_blk_float
        total_mask = mask_blk_float.sum(dim=-1, keepdim=True) + 1e-8
        U_left = (masked_uncertainty * left_weight).sum(dim=-1) / total_mask.squeeze(-1) + 10
        current_left = int(U_left.item())

        ws = max(0, current_block_start - current_left)
        blk_offset = current_block_start - ws

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :ws],)
        
        past_key_values = new_past_key_values
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1

            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            logits_full = model(x[:, ws:], past_key_values=past_key_values, use_cache=True).logits
            logits = logits_full[:, blk_offset:blk_offset + block_length]
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            
            i += 1

    return x, nfe

def get_window_range(block_start, block_end, window_radius, total_length, prompt_length):

    window_start = block_start - window_radius
    window_end = block_end + window_radius
    window_start = max(0, window_start)
    window_end = min(total_length, window_end)
    
    return window_start, window_end

#_haloblock with dual cache
@torch.no_grad()
def generate_with_dual_cache_haloblock(
    model, prompt, steps=256, gen_length=256, block_length=32, halo_factor=1.0, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    total_length = Lp + gen_length
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        out_full = model(x, use_cache=True, output_hidden_states=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False
        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )
        x = torch.where(transfer_index, x0, x)

        logits_blk_init = out_full.logits[:, s:e, :]

        pos_idx = torch.arange(block_length, device=x.device)
        left_weight = 1.0 - (pos_idx.float() / (block_length - 1))
        right_weight = pos_idx.float() / (block_length - 1)

        probs_blk = torch.softmax(logits_blk_init / 10, dim=-1)
        entropy_blk = torch.special.entr(probs_blk).sum(dim=-1)
        mask_blk_init = (x[:, s:e] == mask_id)
        token_uncertainty = entropy_blk * halo_factor

        mask_blk_float = mask_blk_init.float()
        masked_uncertainty = token_uncertainty * mask_blk_float
        total_mask = mask_blk_float.sum(dim=-1, keepdim=True) + 1e-8
        U_left = (masked_uncertainty * left_weight).sum(dim=-1) / total_mask.squeeze(-1)
        U_right = (masked_uncertainty * right_weight).sum(dim=-1) / total_mask.squeeze(-1)
        U_left, U_right = U_left.item(), U_right.item()
        current_left = int(U_left)
        current_right = int(U_right)
        ws, we = get_window_range_asymmetric(s, e, current_left, current_right, total_length)

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, ws:we] = True

        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            
            logits_window = model(
                x[:, ws:we], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits
            
            block_offset_in_window = s - ws
            logits_blk = logits_window[:, block_offset_in_window:block_offset_in_window+block_length]

            mask_blk = (x[:, s:e] == mask_id)
            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            nfe += 1
    return x, nfe

#_fastdllm
@torch.no_grad()
def generate_with_dual_cache_fastdllm(
    model, prompt, steps=128, gen_length=128, block_length=128, halo_factor=1.0, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, nfe

#_adablock
@torch.no_grad()
def generate_with_dual_cache_adablock(
    model, prompt, steps=128, gen_length=128, block_length=128, halo_factor=1.0, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, factor=None
    ):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    init_block_length = 32

    delimiter_threshold = 0.3
    delimiter_ids=[11,13,198]
    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe = 0

    while generated_length < gen_length: 

        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length
        
        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index_adablock(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1
        # 2nd and later denoising steps in block
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, block_start:block_end] == mask_id)
            block_output = model(x[:, block_start:block_end], past_key_values=full_cache, use_cache=True, replace_position=replace_position)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index_adablock(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x[:, block_start:block_end], None, threshold)
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]

    return x, nfe

#_halo_adablock
@torch.no_grad()
def generate_with_dual_cache_halo_adablock(
    model, prompt, steps=128, gen_length=128, block_length=128, halo_factor=1.0, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, factor=None): 
    '''
    AdaBlock + HaloBlock
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top‑k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
        halo_factor: Halo factor
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top‑1)"

    init_block_length = 32

    delimiter_threshold = 0.3
    delimiter_ids=[11,13,198]

    generated_length = 0
    Lp = prompt.shape[1]
    total_length = Lp + gen_length

    nfe = 0
    while generated_length < gen_length: 
        
        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1
        
        block_length = compute_block_length(logits, predicted_tokens, prompt, gen_length, generated_length, init_block_length, delimiter_ids=delimiter_ids, delimiter_threshold=delimiter_threshold)
        
        block_start = Lp + generated_length
        block_end = block_start + block_length
        generated_length += block_length
        
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0
        
        x0, transfer_index = get_transfer_index_adablock(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        logits_blk_init = output.logits[:, block_start:block_end, :]
        blk_len = block_length
        pos_idx = torch.arange(blk_len, device=x.device)
        left_weight = 1.0 - (pos_idx.float() / (blk_len - 1)) if blk_len > 1 else torch.ones_like(pos_idx.float())
        right_weight = pos_idx.float() / (blk_len - 1) if blk_len > 1 else torch.zeros_like(pos_idx.float())

        probs_blk = torch.softmax(logits_blk_init / 10, dim=-1)
        entropy_blk = torch.special.entr(probs_blk).sum(dim=-1)
        mask_blk_init = (x[:, block_start:block_end] == mask_id)

        token_uncertainty = entropy_blk * halo_factor
        mask_blk_float = mask_blk_init.float()
        masked_uncertainty = token_uncertainty * mask_blk_float

        total_mask = mask_blk_float.sum(dim=-1, keepdim=True) + 1e-8
        U_left = (masked_uncertainty * left_weight).sum(dim=-1) / total_mask.squeeze(-1)
        U_right = (masked_uncertainty * right_weight).sum(dim=-1) / total_mask.squeeze(-1)
        U_left, U_right = U_left.item(), U_right.item()

        current_left = int(U_left)
        current_right = int(U_right)
        ws, we = get_window_range_asymmetric(block_start, block_end, current_left, current_right, total_length)

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, ws:we] = True

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

            block_output = model(
                x[:, ws:we],
                past_key_values=full_cache,
                use_cache=True,
                replace_position=replace_position
            )
            window_logits = block_output.logits
            block_offset_in_window = block_start - ws
            block_logits = window_logits[:, block_offset_in_window : block_offset_in_window + block_length]
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1

            mask_index = (x[:, block_start:block_end] == mask_id)
            x0, transfer_index = get_transfer_index_adablock(
                block_logits, block_predicted_tokens, remasking, mask_index, 
                x[:, block_start:block_end], None, threshold
            )
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]

    return x, nfe

# _dparallel
@ torch.no_grad()
def generate_with_dual_cache_dparallel(model, prompt, steps=128, gen_length=128, block_length=128, halo_factor=1.0, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_dparallel(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index_entropy(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            x0, transfer_index = get_transfer_index_entropy(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1

    return x, nfe

#_halo_dparallel
@torch.no_grad()
def generate_with_dual_cache_halo_dparallel(model, prompt, steps=128, gen_length=128, block_length=128, halo_factor=1.0, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        halo_factor: Halo region scaling factor. 0 disables halo (original dual cache behavior).
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_dparallel(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index_entropy(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        if halo_factor > 0:
            logits_blk_init = output.logits[:, current_block_start:current_block_end, :]
            p_blk = F.softmax(logits_blk_init / 2, dim=-1)
            entropy_blk = -torch.sum(p_blk * torch.log(p_blk + 1e-12), dim=-1)

            pos_idx = torch.arange(block_length, device=x.device)
            left_weight = 1.0 - (pos_idx.float() / (block_length - 1))
            right_weight = pos_idx.float() / (block_length - 1)

            mask_blk_init = (x[:, current_block_start:current_block_end] == mask_id)
            mask_blk_float = mask_blk_init.float()
            masked_uncertainty = entropy_blk * halo_factor * mask_blk_float

            total_mask = mask_blk_float.sum(dim=-1, keepdim=True) + 1e-8
            U_left = (masked_uncertainty * left_weight).sum(dim=-1) / total_mask.squeeze(-1)
            U_right = (masked_uncertainty * right_weight).sum(dim=-1) / total_mask.squeeze(-1)

            current_left = int(U_left.item())
            current_right = int(U_right.item())
            ws, we = get_window_range_asymmetric(
                current_block_start, current_block_end,
                current_left, current_right, x.shape[1]
            )
        else:
            ws, we = current_block_start, current_block_end

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, ws:we] = True
        while True:
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            
            logits_window = model(
                x[:, ws:we], 
                past_key_values=past_key_values, 
                use_cache=True, 
                replace_position=replace_position
            ).logits
            
            block_offset = current_block_start - ws
            logits = logits_window[:, block_offset:block_offset + block_length]

            x0, transfer_index = get_transfer_index_entropy(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1
    return x, nfe

#_d3llm
@torch.no_grad()
def generate_with_dual_cache_d3llm(
    model, prompt, steps=128, gen_length=128, block_length=128, halo_factor=1.0, eos_token_id=None, temperature=0.,
    remasking='low_confidence', mask_id=126336, threshold=None, block_add_threshold=0.1, decoded_token_threshold=0.95,
    cache_delay_iter=1, refresh_interval=10000
):
    """
    Pipelined parallel decoding with Delayed KV-Cache.
    
    Strategy: Wait cache_delay_iter steps after a block is fully decoded before caching it.
    This allows token representations to stabilize in the diffusion process.
    """
    max_new_tokens = 256
    block_size = block_length
    early_stop=True

    # Only use eos_token_id if early_stop is enabled
    if not early_stop:
        eos_token_id = None
    x = torch.full((1, prompt.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    
    prompt_length = prompt.shape[1]
    num_blocks = max_new_tokens // block_size
    
    # Track block states with delayed caching support
    block_states = {
        0: {
            "start": 0,
            "end": prompt_length,
            "mask_count": 0,
            "total_masks": prompt_length,
            "is_complete": True,
            "completed_at_nfe": 0,
            "is_cached": False,
        }
    }
    
    # Create first generation block
    next_block_id = 1
    if next_block_id <= num_blocks:
        block_start = prompt_length + (next_block_id - 1) * block_size
        block_end = min(block_start + block_size, prompt_length + max_new_tokens)
        should_activate = 1.0 >= decoded_token_threshold
        block_states[next_block_id] = {
            "start": block_start,
            "end": block_end,
            "mask_count": block_end - block_start,
            "total_masks": block_end - block_start,
            "is_complete": should_activate,
            "completed_at_nfe": None,
            "is_cached": False,
        }
        next_block_id += 1
    
    past_key_values = None
    manual_cache_length = 0
    nfe = 0
    has_eos = False
    
    def update_block_activation_states():
        """Update which blocks should be fully activated based on previous block progress."""
        for bid in sorted(block_states.keys()):
            if bid > 0 and not block_states[bid]["is_complete"]:
                prev_progress = 1 - block_states[bid - 1]["mask_count"] / block_states[bid - 1]["total_masks"]
                if prev_progress >= decoded_token_threshold:
                    block_states[bid]["is_complete"] = True
    
    while True:
        # Check termination
        mask_index = x == mask_id
        total_masks = mask_index[:, prompt_length:].sum()
        
        if total_masks == 0 and next_block_id > num_blocks:
            break
        
        nfe += 1
        
        # Early stop: handle EOS tokens
        if early_stop and eos_token_id is not None:
            has_eos, first_eos_abs = handle_early_stop(x, block_states, eos_token_id, prompt_length,
                                                        mask_token_id=mask_id, debug=False)
            if has_eos:
                # Create all missing blocks after EOS and mark them as complete
                while next_block_id <= num_blocks:
                    block_start = prompt_length + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                    if block_start > first_eos_abs:
                        block_states[next_block_id] = {
                            "start": block_start, "end": block_end, "mask_count": 0,
                            "total_masks": block_end - block_start, "is_complete": True,
                            "completed_at_nfe": nfe, "is_cached": True,
                        }
                        next_block_id += 1
                    else:
                        break
                # Recalculate total_masks
                total_masks = (x == mask_id)[:, prompt_length:].sum()
                if total_masks == 0:
                    break
        
        # Update block activation states
        update_block_activation_states()
        
        # Add new block dynamically based on last block's progress (skip if EOS detected)
        if next_block_id <= num_blocks and not has_eos:
            last_bid = max(block_states.keys())
            if last_bid > 0:
                last_progress = 1 - block_states[last_bid]["mask_count"] / block_states[last_bid]["total_masks"]
                should_add_block = (last_progress >= block_add_threshold) or (block_states[last_bid]["mask_count"] == 0)
                
                if should_add_block:
                    block_start = prompt_length + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                    if block_end > block_start:
                        actual_mask_count = (x[:, block_start:block_end] == mask_id).sum().item()
                        prev_progress = 1 - block_states[next_block_id - 1]["mask_count"] / block_states[next_block_id - 1]["total_masks"]
                        should_activate = prev_progress >= decoded_token_threshold
                        
                        block_states[next_block_id] = {
                            "start": block_start,
                            "end": block_end,
                            "mask_count": actual_mask_count,
                            "total_masks": block_end - block_start,
                            "is_complete": should_activate,
                            "completed_at_nfe": None,
                            "is_cached": False,
                        }
                        next_block_id += 1
        
        # Find rightmost active block
        rightmost_active_bid = 0
        for bid in sorted(block_states.keys()):
            if block_states[bid]["is_complete"] or block_states[bid]["mask_count"] > 0:
                rightmost_active_bid = bid
        
        if rightmost_active_bid == 0:
            break
        
        active_end = block_states[rightmost_active_bid]["end"]
        
        refresh_output = None

        cache_length = manual_cache_length
        
        # Update completion timestamps and find blocks ready to cache
        blocks_to_cache = []
        for bid in sorted(block_states.keys()):
            if block_states[bid]["end"] <= cache_length:
                continue  # Already cached
            
            if block_states[bid]["mask_count"] == 0:  # Fully decoded
                if block_states[bid]["completed_at_nfe"] is None:
                    block_states[bid]["completed_at_nfe"] = nfe
                
                delay = nfe - block_states[bid]["completed_at_nfe"]
                if delay >= cache_delay_iter and not block_states[bid]["is_cached"]:
                    blocks_to_cache.append(bid)
                    block_states[bid]["is_cached"] = True
                elif delay < cache_delay_iter:
                    # This block is stabilizing, cannot cache subsequent blocks
                    break
            else:
                break  # Stop at first incomplete block
        
        # Determine update_kvcache: how many new tokens to add to cache
        update_kvcache = 0
        if blocks_to_cache:
            latest_to_cache = max(blocks_to_cache)
            update_kvcache = block_states[latest_to_cache]["end"] - cache_length
        
        # Determine forward strategy
        # Find the earliest position that needs to be forwarded (not cached)
        forward_start_pos = cache_length  # Start from end of cached region
        
        # Check if there are completed but not cached blocks (in stabilization period)
        has_stabilizing_blocks = False
        for bid in sorted(block_states.keys()):
            if (block_states[bid]["mask_count"] == 0 and 
                block_states[bid]["completed_at_nfe"] is not None and
                not block_states[bid]["is_cached"]):
                # This block is completed but not cached (stabilizing)
                has_stabilizing_blocks = True
                forward_start_pos = min(forward_start_pos, block_states[bid]["start"])
        
        # Also include active blocks
        for bid in sorted(block_states.keys()):
            if block_states[bid]["mask_count"] > 0:
                forward_start_pos = min(forward_start_pos, block_states[bid]["start"])
        
        # If forward_start_pos >= active_end, no blocks need forwarding
        if forward_start_pos >= active_end:
            break
        
        # Determine forward strategy
        if update_kvcache > 0:
            # Need to cache new blocks: forward from cache_length to end
            input_seq = x[:, cache_length:]
            process_start_pos = cache_length
            use_kv_cache = True
        elif past_key_values is not None and not has_stabilizing_blocks and nfe % refresh_interval != 0:
            # Have cache and no stabilizing blocks: can use cache
            input_seq = x[:, forward_start_pos:]
            process_start_pos = forward_start_pos
            use_kv_cache = True
        elif past_key_values is not None and (has_stabilizing_blocks or nfe % refresh_interval == 0):
            # Refresh cache when: 1) has stabilizing blocks, or 2) periodic refresh interval
            # print(f"[DEBUG-KV] has_stabilizing_blocks={has_stabilizing_blocks}, nfe={nfe}, refresh_interval={refresh_interval}")
            if has_stabilizing_blocks:
                latest_stabilizing_bid = max([bid for bid in block_states.keys()
                                             if block_states[bid]["mask_count"] == 0 and 
                                                block_states[bid]["completed_at_nfe"] is not None and
                                                not block_states[bid]["is_cached"]])
                refresh_pos = block_states[latest_stabilizing_bid]["start"]
            else:
                # Periodic refresh: refresh up to cache_length
                refresh_pos = cache_length
            
            # Full forward to get fresh cache (only once for all completed blocks)
            refresh_output = model(x, use_cache=True)
            refresh_cache = refresh_output.past_key_values
            
            # Truncate cache to refresh_pos (mimics prefix_cache strategy)
            past_key_values = DynamicCache()
            for layer_idx in range(len(refresh_cache)):
                k, v = refresh_cache[layer_idx]
                past_key_values.update(k[:, :, :refresh_pos, :], v[:, :, :refresh_pos, :], layer_idx)
            
            input_seq = x
            process_start_pos = 0
            use_kv_cache = False
        else:
            # No cache OR has stabilizing blocks: forward full sequence without cache
            input_seq = x
            process_start_pos = 0
            use_kv_cache = False
        
        stabilizing_blocks = []
        for bid in sorted(block_states.keys()):
            if block_states[bid]["mask_count"] == 0 and block_states[bid]["completed_at_nfe"] is not None:
                if not block_states[bid]["is_cached"]:
                    stabilizing_blocks.append(bid)
                    
        # Forward pass: reuse refresh_output if possible to avoid redundant forward
        if refresh_output is not None and process_start_pos == 0:
            # Refresh forward already computed full sequence, reuse it
            outputs = refresh_output
            logits = refresh_output.logits
        elif use_kv_cache:
            # Pass cache_position for correct positional encoding
            cache_position = torch.arange(
                manual_cache_length, 
                manual_cache_length + input_seq.shape[1], 
                device=model.device
            )
            outputs = model(
                input_seq,
                past_key_values=past_key_values,
                use_cache=True,
                update_kvcache=update_kvcache if update_kvcache > 0 else 0,
                cache_position=cache_position
            )
            logits = outputs.logits
        else:
            outputs = model(input_seq)
            logits = outputs.logits
        
        # Update cache and manual cache length only when we have blocks to cache
        if update_kvcache > 0:
            past_key_values = outputs.past_key_values
            manual_cache_length += update_kvcache
        
        # Map logits to full sequence
        logits_full = torch.full((1, x.shape[1], logits.shape[-1]), 
                                 float('-inf'), dtype=logits.dtype, device=logits.device)
        logits_end = process_start_pos + logits.shape[1]
        logits_full[:, process_start_pos:logits_end, :] = logits
        
        # Mask out future blocks
        mask_index_for_decode = mask_index.clone()
        mask_index_for_decode[:, active_end:] = 0
        
        # Decode with entropy threshold
        x0, transfer_index = get_transfer_index_entropy(
            logits_full,
            temperature,
            remasking,
            mask_index_for_decode,
            x,
            None,
            threshold if threshold is not None else 999.0,
        )
        
        # Ensure progress in fully activated blocks
        first_fully_activated_bid = None
        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                first_fully_activated_bid = bid
                break
        
        if first_fully_activated_bid is not None:
            start, end = block_states[first_fully_activated_bid]["start"], block_states[first_fully_activated_bid]["end"]
            block_transfer = transfer_index[:, start:end]
            
            if not block_transfer.any():
                # Force decode the lowest entropy token
                p = F.softmax(logits_full[:, start:end].to(torch.float64), dim=-1)
                entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
                block_mask = mask_index_for_decode[:, start:end]
                entropy = torch.where(block_mask, entropy, torch.inf)
                best_idx = entropy[0].argmin()
                transfer_index[0, start + best_idx] = True
                x0_resample = torch.argmax(logits_full[0, start + best_idx], dim=-1)
                x0[0, start + best_idx] = x0_resample
        
        # Apply decoded tokens
        x[transfer_index] = x0[transfer_index]
        
        # Update block mask counts
        for bid in sorted(block_states.keys()):
            if bid > 0:
                start, end = block_states[bid]["start"], block_states[bid]["end"]
                block_decoded = transfer_index[:, start:end].sum().item()
                if block_decoded > 0:
                    block_states[bid]["mask_count"] = max(0, block_states[bid]["mask_count"] - block_decoded)
        
        if nfe > 10000:
            break
    
    return x, nfe

#_dparallel
def get_transfer_index_dparallel(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

#_fastdllm
def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

#_adablock
def get_transfer_index_adablock(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    x0 = predicted_tokens  # (B, L)

    # Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)  # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
