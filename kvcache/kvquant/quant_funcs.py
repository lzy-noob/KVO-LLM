import torch
import copy
'''This funcs implement the dfkv quantization method'''

class KVQuantizer(torch.nn.Module):
    def __init__(self, args: dict) -> None:
        
        super().__init__()
        # quant params
        self.n_bits             = args['n_bits']
        self.sym                = args['symmetric']
        self.clip_ratio         = args['clip_ratio']
        # kivi parameter
        self.group_size         = args['group_size']
        # adv params
        self.quant_grain        = args['quant_grain']

        # quant cache
        self.diff_len           = 0 
        self.chunk_size         = args['chunk_size']

        # prune
        self.prune_en           = args['prune_en']
        self.prune_budget       = args['prune_budget']

    @torch.no_grad()
    def _reset_quant(self):
        self.diff_len           = 0

    @torch.no_grad()
    def forward(self, feat, diff_len):
        
        if self.quant_grain == 'key':
            feat = self.kvo_key_quantize(
                act         = feat,
                diff_len    = diff_len
            )
        elif self.quant_grain == 'value':
            feat = self.kvo_value_quantize(
                act         = feat,
                diff_len    = diff_len,
            )
        else:
            assert False, "Do not support this quant grain"
        
        return feat

    @torch.no_grad()
    def key_quantizer(self,
                        act: torch.tensor, 
                        n_bits, 
                        group_size,
                        sym, 
                        clip_ratio, 
                        quant_grain,
                        prune_en,
                        prune_budget,
                    ) -> torch.tensor:

        if n_bits >= 16:
            return act, None, None

        assert act.is_contiguous(), "tensor should be continous for bitsandbytes kernel."
        assert act.dim() == 2, "act format should be: [token, hid_dim]"

        # group num
        assert act.shape[-1] % group_size == 0, "act shape is not mode group size"
        channel_group_num = act.shape[-1] // group_size
        
        # prune num
        channel_prune_budget = int(group_size * (1 - prune_budget))
        # quant max min
        if sym:
            q_max = (2**(n_bits-1)-1)
            q_min = (-2**(n_bits-1)) 
        else:
            q_max = (2**(n_bits)-1)
            q_min = (0) 
        
        act       = act.reshape(
            act.shape[0],
            channel_group_num,
            group_size,
        )
        act_max   = act.amax(dim=-1,keepdim=True)
        act_min   = act.amin(dim=-1,keepdim=True)

        # generate scale
        if sym:
            if clip_ratio < 1.0:
                act_max = act_max * clip_ratio
            scales = act_max / q_max
            scales = scales.clamp(min=1e-5)
            base = torch.zeros_like(scales)
        else:
            if clip_ratio < 1.0:
                act_max *= clip_ratio
                act_min *= clip_ratio
            scales = (act_max-act_min) / q_max
            scales = scales.clamp(min=1e-5)
            base   = act_min

        # fake quant procedure
        act_int   = torch.clamp(torch.round((act-base) / scales), q_min, q_max)
        # act       = (act_int - base) * scales
        act       = act_int * scales + base
        
        # prune func
        if prune_en:
            avg_sum = abs(act)
            _,topk = avg_sum.topk(
                k=channel_prune_budget, dim=-1, largest=False, sorted=False
            ) # topk [token, group_num, topk]
            act = act.scatter(-1, topk, 0)

        # reshape
        act       = act.reshape(act.shape[0],-1)

        return act, scales, base

    @torch.no_grad()
    def diff_key_quantize(  self,
                            act: torch.tensor, 
                            n_bits, 
                            group_size,
                            sym, 
                            clip_ratio, 
                            quant_grain,
                            chunk_size,
                        ) -> torch.tensor:


        assert act.dim() == 2, "act format should be: [num_groups, group_size]"

        # trunk num
        assert chunk_size > 0, "trunk size should > 0"
        if act.shape[0] % chunk_size == 0:
            chunk_group_num = act.shape[0] // chunk_size
        else: 
            assert False, "chunk_group_num should be integer"

        # generate base matrix [group, 1, hidden]
        act = act.reshape(
            chunk_group_num,
            chunk_size,
            act.shape[-1]
        )
        act_base = act[:,0,:].clone().contiguous()
        # quant base matrix into int 8
        act_base,_,_ = self.key_quantizer(
            act         = act_base,
            n_bits      = 8,
            group_size  = group_size,
            sym         = sym,
            clip_ratio  = clip_ratio,
            quant_grain = quant_grain, 
            prune_en    = False,
            prune_budget= 0,
        )
        # reshape act
        act_base        = act_base.unsqueeze(dim=1)
        act_diff        = act - act_base
        act_diff[:,0,:] = 0 
        act_diff        = act_diff.reshape(
            -1,
            act_diff.shape[-1]
        )
        # generate diff matrix
        act_diff,_,_ = self.key_quantizer(
            act         = act_diff,
            n_bits      = n_bits,
            group_size  = group_size,
            sym         = sym,
            clip_ratio  = clip_ratio,
            quant_grain = quant_grain, 
            prune_en    = self.prune_en,
            prune_budget= self.prune_budget,
        )
        # reshaoe act_diff
        act_diff = act_diff.reshape(
            chunk_group_num,
            chunk_size,
            act_diff.shape[-1]
        )
        # merge act_diff and act_base 
        act = act_base + act_diff
        act = act.reshape(
            -1,
            act.shape[-1]
        )
        del act_diff, act_base

        return act

    @torch.no_grad()
    def kvo_key_quantize(self, act: torch.tensor, diff_len: int) -> torch.tensor:
        
        act = act.squeeze(dim=0).permute(1,0,2)
        savedShape = act.shape
        act = act.reshape(savedShape[0],-1)

        assert act.dim() == 2, "act format should be: [token, hid_dim]"
        assert act.is_contiguous(), "tensor should be continous for bitsandbytes kernel."   

        # figure out the diff len
        assert diff_len >= self.diff_len, "Error, diff_len > self.diff_len, this will never happend"
        if diff_len == self.diff_len:
            act = act.reshape(savedShape)
            act = act.permute(1,0,2).unsqueeze(dim=0)   
            return act
        else:
            quant_len = diff_len - self.diff_len   
        # act partitation
        quant_act = act[self.diff_len:diff_len,:]
        # act quantization
        quant_act = self.diff_key_quantize(
            act         = quant_act,
            n_bits      = self.n_bits,
            group_size  = self.group_size,
            sym         = self.sym,
            clip_ratio  = self.clip_ratio,
            quant_grain = "per_group", 
            chunk_size  = self.chunk_size,
        )

        # update act
        act[self.diff_len:diff_len,:] = quant_act

        # update parameter
        self.diff_len = diff_len
        
        # reshape
        act = act.reshape(savedShape)
        act = act.permute(1,0,2).unsqueeze(dim=0)

        return act
    
    @torch.no_grad()
    def value_quantizer(self,
                        act: torch.tensor, 
                        n_bits, 
                        chunk_size,
                        sym, 
                        clip_ratio, 
                        prune_en,
                        prune_budget,
                    ) -> torch.tensor:

        if n_bits >= 16:
            return act, None, None

        assert act.is_contiguous(), "tensor should be continous for bitsandbytes kernel."
        assert act.dim() == 2, "act format should be: [-1, group_size]"

        # [chunk_group_num x hid_dim, group size]
        assert act.shape[-1] % chunk_size == 0

        # prune num
        if prune_en:
            channel_prune_budget = int(act.shape[-1] * (1 - prune_budget))

        # find act base
        chunk_num   = act.shape[-1] // chunk_size 
    
        act_base    = act[:,::chunk_size].clone()
        assert act_base.shape[-1] == chunk_num 

        # mask act base
        act[:,::chunk_size] = 0

        # quant max min
        if sym:
            q_max       = (2**(n_bits-1)-1)
            q_min       = (-2**(n_bits-1)) 
            double_qmax = (2 ** (2 * n_bits-1)-1)
            double_qmin = (-2 ** (2 * n_bits-1))
        else:
            q_max       = (2**(n_bits)-1)
            q_min       = (0) 
            double_qmax = (2 ** (2 * n_bits)-1)
            double_qmin = (0)

        act_max   = act.amax(dim=-1,keepdim=True)
        act_min   = act.amin(dim=-1,keepdim=True)

        # generate scale
        if sym:
            if clip_ratio < 1.0:
                act_max = act_max * clip_ratio
            scales = act_max / q_max
            scales = scales.clamp(min=1e-5)
            base = torch.zeros_like(scales)
        else:
            if clip_ratio < 1.0:
                act_max *= clip_ratio
                act_min *= clip_ratio
            scales = (act_max-act_min) / q_max
            scales = scales.clamp(min=1e-5)
            base   = act_min

        # fake quant procedure
        act_int     = torch.clamp(torch.round((act-base) / scales), q_min, q_max)
        act         = act_int * scales + base
        
        # mask act base
        act[:,::chunk_size] = 0

        # prune func
        if prune_en:
            avg_sum = abs(act)
            _,topk = avg_sum.topk(
                k=channel_prune_budget + chunk_num, dim=-1, largest=False, sorted=False
            ) # topk [-1, topk]
            act = act.scatter(-1, topk, 0)

        # generate act base scale
        scales              = scales * q_max / double_qmax
        scales              = scales.clamp(min=1e-5)

        act_base_int        = torch.clamp(torch.round((act_base-base) / scales), double_qmin, double_qmax)
        act_base            = act_base_int * scales + base
        
        # update act_base
        act[:,::chunk_size] = act_base 

        return act, scales, base



    @torch.no_grad()
    def diff_value_quantize(    self,
                                act: torch.tensor, 
                                n_bits, 
                                group_size,
                                sym, 
                                clip_ratio, 
                                quant_grain,
                                chunk_size,
                            ) -> torch.tensor:

        assert act.dim() == 2, "act format should be: [tokens, hid dim]"

        # group size
        assert group_size > 0, "group_size should > 0"
        if act.shape[0] % group_size == 0:
            chunk_group_num = act.shape[0] // group_size
        else: 
            assert False, "chunk_group_num should be integer"

        # generate matrix [chunk_group_num, group_size, hid_dim]
        act = act.reshape(
            chunk_group_num,
            group_size,
            act.shape[-1],
        )
        # [group size, chunk_group_num,  hid_dim]
        act = act.permute(1,0,2)
        # [group size, chunk_group_num x hid_dim]
        act = act.reshape(group_size,-1)
        # [chunk_group_num x hid_dim, group size]
        act = act.transpose(1,0).contiguous()

        # quantization act
        act,_,_ = self.value_quantizer(
            act         = act,
            n_bits      = n_bits,
            chunk_size  = chunk_size,
            sym         = sym,
            clip_ratio  = clip_ratio,
            prune_en    = self.prune_en,
            prune_budget= self.prune_budget,
        )
        # reshape act
        # [group size, chunk_group_num x hid_dim]
        act = act.transpose(1,0).contiguous()
        # [group size, chunk_group_num, hid_dim]
        act = act.reshape(
            group_size,
            chunk_group_num,
            -1
        )
        # [chunk_group_num, group size, hid_dim]
        act = act.permute(1,0,2)

        # update act
        act = act.reshape(
            -1,
            act.shape[-1]
        )

        return act

    @torch.no_grad()
    def kvo_value_quantize(self, act: torch.tensor, diff_len: int) -> torch.tensor:

        act = act.squeeze(dim=0).permute(1,0,2)
        savedShape = act.shape
        act = act.reshape(savedShape[0],-1)

        assert act.dim() == 2, "act format should be: [token, hid_dim]"
        assert act.is_contiguous(), "tensor should be continous for bitsandbytes kernel."
        
        # figure out the diff len
        assert diff_len >= self.diff_len, "Error, diff_len > self.diff_len, this will never happend"
        if diff_len == self.diff_len:
            act = act.reshape(savedShape)
            act = act.permute(1,0,2).unsqueeze(dim=0)   
            return act
        else:
            quant_len = diff_len - self.diff_len

        # act partitation
        quant_act = act[self.diff_len:diff_len,:]
        # act quantization
        quant_act = self.diff_value_quantize(
            act         = quant_act,
            n_bits      = self.n_bits,
            group_size  = self.group_size,
            sym         = self.sym,
            clip_ratio  = self.clip_ratio,
            quant_grain = "per_group", 
            chunk_size  = self.chunk_size,
        )  
        
        # update act
        act[self.diff_len:diff_len,:] = quant_act

        # update parameter
        self.diff_len = diff_len
        
        # reshape
        act = act.reshape(savedShape)
        act = act.permute(1,0,2).unsqueeze(dim=0)

        return act
    