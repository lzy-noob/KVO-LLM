import torch
import copy

'''This funcs implement the normal quantization method'''
class KVQuantizer(torch.nn.Module):
    def __init__(self, args: dict) -> None:
        
        super().__init__()

        # quant params
        self.n_bits             = args['n_bits']
        self.sym                = args['symmetric']
        self.group_size         = args['group_size']
        self.clip_ratio         = args['clip_ratio']

    @torch.no_grad()
    def forward(self, feat):
        
        feat,_,_ = self.normal_quantize(
            act         = feat,
            n_bits      = self.n_bits,
            group_size  = self.group_size,
            sym         = self.sym,
            clip_ratio  = self.clip_ratio, 
        )

        return feat

    @torch.no_grad()
    def normal_quantize(self,
                        act: torch.tensor, 
                        n_bits, 
                        group_size,
                        sym, 
                        clip_ratio,
                    ) -> torch.tensor:

        if n_bits >= 16:
            return act, None, None

        savedDim = act.dim()
        if savedDim != 2:
            assert savedDim == 4, "Unexpected input dim!"
            act = act.squeeze(dim=0).permute(1,0,2)

        savedShape = act.shape
        act = act.reshape(savedShape[0],-1)

        assert act.is_contiguous(), "tensor should be continous for bitsandbytes kernel."
        assert act.dim() == 2, "act format should be: [num_groups, group_size]"

        # group num
        assert act.shape[-1] % group_size == 0, "act shape is not mode group size"
        channel_group_num = act.shape[-1] // group_size

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
        act       = act.reshape(act.shape[0],-1)

        act       = act.reshape(savedShape)
        if savedDim != 2:
            act       = act.permute(1,0,2).unsqueeze(dim=0)

        return act, scales, base
