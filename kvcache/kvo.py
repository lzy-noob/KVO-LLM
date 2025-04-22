import math
import warnings
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint

import types


from kvcache.normalquant.quant_funcs import KVQuantizer as normalquantizer
from kvcache.kvquant.quant_funcs import KVQuantizer as kvquantizer

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      apply_rotary_pos_emb,
                                                      LlamaForCausalLM,
                                                      repeat_kv)

__all__ = ['convert_kvcache_llama_kvo',
           'LlamaAttention_kvo',
           'LlamaForCausalLMKVO']

def prompt_attn_weight_mask(attn_weights, window_size, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expand attn weights to mod window size
    seq_length      = attn_weights.shape[-1]
    padding_length  = window_size - ((seq_length - 1) % window_size + 1)
    if padding_length > 0:
        attn_weights    = torch.cat(
            [
                attn_weights,
                torch.zeros(
                    (
                        attn_weights.shape[0],
                        attn_weights.shape[1],
                        attn_weights.shape[2],
                        padding_length,
                    ),
                    device=attn_weights.device,
                )
            ],
            dim = -1,
        )
        attn_weights   = torch.cat(
            [
                attn_weights,
                torch.zeros(
                    (
                        attn_weights.shape[0],
                        attn_weights.shape[1],
                        padding_length,
                        attn_weights.shape[3],
                    ),
                    device=attn_weights.device,
                )
            ],
            dim = -2,
        )
    assert (attn_weights.shape[-1] % window_size) == 0, "attn weights shape error"
    assert (attn_weights.shape[-2] % window_size) == 0, "attn weights shape error"
    # window attn_weights into window size tokens
    window_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2] // window_size,
        window_size,
        attn_weights.shape[3],
    ) # [BS, head, window_group, window_size, expand_size]
    window_group = attn_weights.shape[2] // window_size
    window_attn_weights_sum = window_attn_weights.sum(dim=-2)
    # [BS, head, window_group-1, expand_size]
    window_attn_weights_sum = window_attn_weights_sum[:,:,:-1,:] + window_attn_weights_sum[:,:,1:,:]
    
    # select vld window
    if padding_length > 0 :
        window_vld_num = window_group - 2 
    else:
        window_vld_num = window_group - 1

    # cur processing window weights [BS, head, window_vld_num * window_size]
    window_vld_attn_weights = torch.zeros(
        (
            attn_weights.shape[0],
            attn_weights.shape[1],
            window_vld_num * window_size
        ),
        device=attn_weights.device,
    )

    for i in range(window_vld_num):
        window_vld_attn_weights[:,:,i*window_size:(i+1)*window_size] = window_attn_weights_sum[:,:,i,i*window_size:(i+1)*window_size] 
    # window vld attn weights [BS, head, chunk_num, chunk_size]
    chunk_num = window_vld_num * window_size // chunk_size
    window_vld_attn_weights = window_vld_attn_weights.reshape(
        window_vld_attn_weights.shape[0],
        window_vld_attn_weights.shape[1],
        chunk_num,
        chunk_size,
    )
    # sort chunk base on attn weights
    indices     = torch.argsort(window_vld_attn_weights,dim=-1,descending=True)
    idx_base    = torch.arange(chunk_num, device=indices.device) * chunk_size
    idx_base    = idx_base.unsqueeze(-1).repeat(1, chunk_size)
    
    # generate indices for token
    indices     = indices + idx_base
    indices     = indices.reshape(
        indices.shape[0],
        indices.shape[1],
        -1
    )
    # [BS, head, window_vld_num * window_size]
    cur_differential_len = indices.shape[-1]
    cur_attention_sum    = torch.zeros(
        (
            attn_weights.shape[0],
            attn_weights.shape[1],
            2 * window_size
        ),
        device=attn_weights.device,
    )
        
    if padding_length > 0:
        cur_attention_sum = window_attn_weights_sum[:,:,-1,-2*window_size:]
    else:
        cur_attention_sum[:,:,0:window_size] = window_attn_weights_sum[:,:,-1,-window_size:]

    # padding indices [BS, head, query]
    indices_append_len  = seq_length - cur_differential_len
    indices_append      = torch.arange(indices_append_len, device=attn_weights.device) + cur_differential_len
    indices_append      = indices_append.unsqueeze(dim=0).unsqueeze(dim=0)
    indices_append      = indices_append.repeat(
        indices.shape[0], indices.shape[1], 1
    )  

    indices             = torch.cat(
        [
            indices,
            indices_append,
        ],
        dim=-1
    )

    return indices, cur_differential_len, cur_attention_sum

def gen_attn_weight_mask(attn_weights, window_size, chunk_size, past_differential_len, past_attention_sum):
    # attn_weights (BS, head, 1, keys)

    # expand attn weights to mod window size
    seq_length      = attn_weights.shape[-1]
    padding_length  = window_size - ((seq_length - 1) % window_size + 1)
    if padding_length > 0:
        attn_weights    = torch.cat(
            [
                attn_weights,
                torch.zeros(
                    (
                        attn_weights.shape[0],
                        attn_weights.shape[1],
                        attn_weights.shape[2],
                        padding_length,
                    ),
                    device=attn_weights.device,
                )
            ],
            dim = -1,
        )
    # select attn_weights
    attn_weights    = attn_weights[:,:,:,past_differential_len:]
    assert (attn_weights.shape[-1] % window_size) == 0, "attn weights shape is not mode window size"
    # update attention sum
    if past_attention_sum == None:
        cur_attention_sum = torch.zeros(
            (
                attn_weights.shape[0],
                attn_weights.shape[1],
                2 * window_size
            ),
            device=attn_weights.device
        )
    else:
        cur_attention_sum = past_attention_sum
    window_num          = attn_weights.shape[-1] // window_size
    cur_attention_sum[:,:,0:window_num*window_size] += attn_weights.squeeze(dim=2)
    # update indices and cur differential sum
    indices = None
    cur_differential_len = past_differential_len
    # sort chunk base on attn weights
    if (padding_length == 0) & (window_num == 2):
        # window vld attn weights [BS, head, window_size]
        window_vld_attn_weights = torch.zeros(
            (
                attn_weights.shape[0],
                attn_weights.shape[1],
                window_size
            ),
            device=attn_weights.device,
        )
        window_vld_attn_weights = cur_attention_sum[:,:,0:window_size]

        # window vld attn weights [BS, head, chunk_num, chunk_size]
        chunk_num =  window_size // chunk_size
        window_vld_attn_weights = window_vld_attn_weights.reshape(
            window_vld_attn_weights.shape[0],
            window_vld_attn_weights.shape[1],
            chunk_num,
            chunk_size,
        )

        # sort chunk base on attn weights
        indices     = torch.argsort(window_vld_attn_weights,dim=-1,descending=True)
        idx_base    = torch.arange(chunk_num, device=indices.device) * chunk_size
        idx_base    = idx_base.unsqueeze(-1).repeat(1, chunk_size)
        
        # generate indices for token
        indices     = indices + idx_base
        indices     = indices.reshape(
            indices.shape[0],
            indices.shape[1],
            -1
        )

        # update differential param
        # - cur differential len
        cur_differential_len = past_differential_len + window_size
        # - cur window sum
        cur_attention_sum_padding = torch.zeros(
            (
                cur_attention_sum.shape[0],
                cur_attention_sum.shape[1],
                window_size,
            ),
            device=attn_weights.device
        )
        cur_attention_sum = torch.cat(
            [
                cur_attention_sum,
                cur_attention_sum_padding,
            ],
            dim=-1
        )[:,:,-2*window_size:]
        
    return indices, cur_differential_len, cur_attention_sum

def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom


class LlamaAttention_kvo(LlamaAttention):
    """Multi-headed attention"""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # prune ratio
        self.window_diff_select     = config.kvcache['window_diff_select']
        self.heavy_budget           = config.kvcache['start_budget']
        self.window_size            = config.kvcache['recent_budget']
        self.chunk_size             = config.kvcache['key_quant']['chunk_size']
        if self.window_diff_select:
            assert (self.window_size % self.chunk_size) == 0, "window size is not mode chunk size"

        # Nor quantizer
        key_normal_config = {
            'n_bits':       8,
            'symmetric':    config.kvcache['key_quant']['symmetric'],
            'group_size':   config.kvcache['key_quant']['group_size'],
            'clip_ratio':   config.kvcache['key_quant']['clip_ratio'],
        }
        self.key_normal_quantizer   = normalquantizer(key_normal_config)
        # key parameter
        self.key_quantizer          = kvquantizer(config.kvcache['key_quant'])
        # value parameter
        self.value_quantizer        = kvquantizer(config.kvcache['value_quant'])
        
        # differential parameter
        self.cur_differential_len   = 0
        self.cur_attention_sum      = None    

    def _reset_masks(self):
        self.cur_differential_len = 0
        self.cur_attention_sum    = None
        self.key_quantizer._reset_quant()
        self.value_quantizer._reset_quant()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        if q_len > 1:
            return self.prefilling_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                **kwargs,
            )
        else:
            return self.decoding_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                **kwargs,
            )

    def prefilling_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # INT8 Quantized for Recent Tokens
        key_states      = self.key_normal_quantizer(key_states)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # generate output
        attn_output = torch.matmul(attn_weights, value_states)

        assert self.cur_differential_len == 0, "cur differential len is not zero!"
        # *******************************************************#
        # * key code for window_diff_select
        if self.window_diff_select:

            if attn_weights.shape[-1] >= 2 * self.window_size:

                idx, cur_differential_len, cur_attention_sum = prompt_attn_weight_mask(attn_weights, self.window_size, self.chunk_size)
                
                # reshape to [head * query]
                idx         = idx.squeeze()
                idx_base    = torch.arange(self.num_heads, device=idx.device) * idx.shape[-1]
                idx_base    = idx_base.unsqueeze(dim=-1)
                idx_base    = idx_base.repeat(
                    1, idx.shape[-1]
                )
                idx         = idx + idx_base
                idx         = idx.reshape(-1)

                # reorder states
                key_states      = key_states.squeeze().reshape(-1,key_states.shape[-1])
                value_states    = value_states.squeeze().reshape(-1,value_states.shape[-1])

                key_states      = torch.index_select(key_states, 0, idx)
                value_states    = torch.index_select(value_states, 0, idx) 

                key_states      = key_states.reshape(self.num_heads,-1,key_states.shape[-1]).unsqueeze(dim=0)
                value_states    = value_states.reshape(self.num_heads,-1,value_states.shape[-1]).unsqueeze(dim=0)
                
                # update kv cache
                past_key_value = (key_states, value_states) if use_cache else None

                # update differential parameter
                self.cur_differential_len = cur_differential_len
                self.cur_attention_sum    = cur_attention_sum
        # * end code for window_diff_select
        # *******************************************************#

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



    def decoding_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # INT8 Quantized for Recent Tokens
        key_states      = self.key_normal_quantizer(key_states)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # quant
        if self.window_diff_select:
            key_states      = self.key_quantizer(key_states,self.cur_differential_len)
            value_states    = self.value_quantizer(value_states,self.cur_differential_len)
        else:
            kv_seq_len_before = kv_seq_len - 1 # to be similar with window diff select
            padding_length  = self.window_size - ((kv_seq_len_before - 1) % self.window_size + 1)
            total_length    = kv_seq_len_before + padding_length
            
            if total_length >= 2 * self.window_size:
                if padding_length == 0:
                    self.cur_differential_len = total_length - self.window_size
                else:
                    self.cur_differential_len = total_length - 2 * self.window_size

            key_states      = self.key_quantizer(key_states,self.cur_differential_len)
            value_states    = self.value_quantizer(value_states,self.cur_differential_len)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # *******************************************************#
        # * key code for DFKV
        diff_key = key_states.clone()
        # expend diff_key to be divisible by chunk_size
        seq_length = diff_key.shape[-2]
        padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
        diff_key = torch.cat(
            [
                diff_key,
                torch.ones(
                    (diff_key.shape[0], diff_key.shape[1], padding_length, diff_key.shape[3]),
                    device=diff_key.device,
                )
                * torch.tensor(torch.finfo(diff_key.dtype).min),
            ],
            dim=-2,
        )
        # chunk diff_key into chunk_size tokens
        chunk_diff_key = diff_key.reshape(
            diff_key.shape[0],
            diff_key.shape[1],
            diff_key.shape[2] // self.chunk_size,
            self.chunk_size,
            diff_key.shape[3],
        )
        chunk_diff_key = chunk_diff_key[:,:,:,0,:]
        # duplicate chunk_diff_key chunk_size times
        chunk_diff_key = chunk_diff_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
        # reshape chunk_diff_key to the original shape
        chunk_diff_key = chunk_diff_key.reshape(
            chunk_diff_key.shape[0], chunk_diff_key.shape[1], -1, chunk_diff_key.shape[-1]
        )[:, :, :seq_length, :]

        quantized_weight = torch.matmul(
            query_states.float(),
            chunk_diff_key.transpose(2, 3),
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            quantized_weight = quantized_weight + attention_mask
            quantized_weight = torch.max(
                quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
            )
        # select attention weight to fetch local heavy hitter
        cur_quantized_len   = quantized_weight.shape[-1] - 1
        padding_length      = self.window_size - ((cur_quantized_len - 1) % self.window_size + 1)
        total_length        = cur_quantized_len + padding_length
        if total_length >= 2 * self.window_size:
            if padding_length == 0:
                select_len  = total_length - self.window_size
            else:
                select_len  = total_length - 2 * self.window_size
            recent_len  = quantized_weight.shape[-1] - select_len
            # select weights
            attn_weights_for_selection = quantized_weight[:,:,:,:select_len]
            # token budget            
            token_budget = min(select_len, self.heavy_budget)
            assert token_budget > 0, "token budget should > 0"
            # geenrate mask bottom
            mask_bottom = local_heavy_hitter_mask(
                attn_weights_for_selection, token_budget, self.chunk_size
            )  # Default: No padding applied to input
            mask_bottom_append = torch.ones(
                (
                    mask_bottom.shape[0],
                    mask_bottom.shape[1],
                    mask_bottom.shape[2],
                    recent_len,
                ),
                dtype=torch.bool,
                device=mask_bottom.device
            )
            mask_bottom = torch.cat(
                [
                    mask_bottom,
                    mask_bottom_append,
                ],
                dim=-1
            )
            mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
            attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)
        # *******************************************************#

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)


        # *******************************************************#
        # * key code for window_decode_select
        if self.window_diff_select:
            idx, cur_differential_len, cur_attention_sum = gen_attn_weight_mask(
                attn_weights            =attn_weights, 
                window_size             =self.window_size, 
                chunk_size              =self.chunk_size, 
                past_differential_len   =self.cur_differential_len, 
                past_attention_sum      =self.cur_attention_sum,
            )

            if idx is not None:
                # reshape to [head * query]
                idx         = idx.squeeze()
                idx_base    = torch.arange(self.num_heads, device=idx.device) * idx.shape[-1]
                idx_base    = idx_base.unsqueeze(dim=-1)
                idx_base    = idx_base.repeat(
                    1, idx.shape[-1]
                )
                idx         = idx + idx_base
                idx         = idx.reshape(-1)

                key_states_reorder      = key_states[:,:,-2*self.window_size:-self.window_size]
                value_states_reorder    = value_states[:,:,-2*self.window_size:-self.window_size]

                key_states_reorder      = key_states_reorder.squeeze().reshape(-1,key_states_reorder.shape[-1])
                value_states_reorder    = value_states_reorder.squeeze().reshape(-1,value_states_reorder.shape[-1])

                key_states_reorder      = torch.index_select(key_states_reorder, 0, idx)
                value_states_reorder    = torch.index_select(value_states_reorder, 0, idx)

                key_states_reorder      = key_states_reorder.reshape(self.num_heads,-1,key_states_reorder.shape[-1]).unsqueeze(dim=0)
                value_states_reorder    = value_states_reorder.reshape(self.num_heads,-1,value_states_reorder.shape[-1]).unsqueeze(dim=0)

                key_states[:,:,-2*self.window_size:-self.window_size]   = key_states_reorder
                value_states[:,:,-2*self.window_size:-self.window_size] = value_states_reorder
                # update kv cache
                past_key_value = (key_states, value_states) if use_cache else None

            # update differential parameter
            self.cur_differential_len   = cur_differential_len
            self.cur_attention_sum      = cur_attention_sum
        # * end code for window_decode_select
        # *******************************************************#
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



def convert_kvcache_llama_kvo(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_kvo(module, config)

        if isinstance(module, LlamaAttention):
            model._modules[name] = LlamaAttention_kvo(config)

            target_device = next(module.parameters()).device
            for param in model._modules[name].parameters():
                param.data = param.data.to(target_device)
            for buffer in model._modules[name].buffers():
                buffer.data = buffer.data.to(target_device)
            model._modules[name].half()
            
    return model

class LlamaForCausalLMKVO(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        for layer_i in range(2, len(self.model.layers)):
            self.model.layers[layer_i].self_attn = LlamaAttention_kvo(config)