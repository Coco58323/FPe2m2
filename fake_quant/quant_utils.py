import transformers
import torch
import utils
def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def sym_quant_groupwise(w, groupsize, n_bits=4):
    out_features, in_features = w.size()
    w = w.reshape(-1,groupsize)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    w = w.reshape(out_features, in_features)
    return w

def retain_first_mantissa_bit(tensor):
    int_tensor = tensor.view(torch.int16)
    
    sign_bit = (int_tensor >> 15) & 1
    
    exponent_bits = (int_tensor >> 10) & 0x1F
    first_mantissa_bit = (int_tensor >>9) & 1  
    second_mantissa_bit = (int_tensor >> 8) & 1
    first_and_sec = first_mantissa_bit & second_mantissa_bit
    exponent_bits = torch.where(first_and_sec == 1, exponent_bits+1, exponent_bits)
    first_mantissa_bit = torch.where(first_and_sec == 1, 0, first_mantissa_bit)
    second_mantissa_bit = torch.where(first_and_sec == 1, 0, second_mantissa_bit)
    first_mantissa_bit = first_mantissa_bit | second_mantissa_bit
    mask = torch.where(exponent_bits == 10, first_mantissa_bit, 1)
    exponent_bits = torch.where(mask == 0, 0, exponent_bits)
    exponent_bits = torch.where(exponent_bits < 10, 0, exponent_bits)
    first_mantissa_bit = torch.where(exponent_bits == 0, 0, first_mantissa_bit)
    new_int_tensor = (sign_bit << 15) | (exponent_bits << 10) | (first_mantissa_bit << 9)
    new_tensor = new_int_tensor.view(torch.float16)

    return new_tensor

def retain_first_and_sec_mantissa_bit(tensor):
    int_tensor = tensor.view(torch.int16)
    sign_bit = (int_tensor >> 15) & 1
    exponent_bits = (int_tensor >> 10) & 0x1F

    third_mantissa_bit = (int_tensor >> 7) & 1
    fi_sec_thi = (int_tensor >> 7) & 0x7
    exponent_bits = torch.where(fi_sec_thi == 7, exponent_bits+1, exponent_bits)
    fi_sec = (int_tensor>>8) & 0x3
    fi_sec = torch.where(third_mantissa_bit == 1, fi_sec+1, fi_sec)
    fi_sec = torch.where(fi_sec == 4, 0, fi_sec)

    mask = torch.where(exponent_bits == 15, fi_sec, 1)
    exponent_bits = torch.where(mask == 0, 0, exponent_bits)
    exponent_bits = torch.where(exponent_bits<15, 0, exponent_bits)
    first_mantissa_bit = fi_sec << 8
    new_int_tensor = (sign_bit << 15) | (exponent_bits << 10) | first_mantissa_bit
    new_tensor = new_int_tensor.view(torch.float16)
    return new_tensor

def retain_zero_mantissa_bit(tensor):
    int_tensor = tensor.view(torch.int16)
    sign_bit = (int_tensor >> 15) & 1
    exponent_bits = (int_tensor >> 10) & 0x1F

    first_mantissa_bit = (int_tensor >> 9) & 1
    exponent_bits = torch.where(first_mantissa_bit == 1, exponent_bits+1, exponent_bits)
    exponent_bits = torch.where(exponent_bits<9, 0, exponent_bits)

    new_int_tensor = (sign_bit << 15) | (exponent_bits << 10) 
    new_tensor = new_int_tensor.view(torch.float16)
    return new_tensor

def sym_quant_fpe4m0_real(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 16
    scales.div_(q_max).clamp_(min=1e-5)
    w = w.div(scales)

    w_sim=retain_zero_mantissa_bit(w)

    w_sim = w_sim.mul(scales)
    w_sim = w_sim.reshape(w_shape)
    return w_sim

#usage: w = sym_quant_fpe2m2_real(w,-1)
def sym_quant_fpe2m2_real(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 14
    scales.div_(q_max).clamp_(min=1e-5)
    w = w.div(scales)

    w_sim=retain_first_and_sec_mantissa_bit(w)

    w_sim = w_sim.mul(scales)
    w_sim = w_sim.reshape(w_shape)
    return w_sim

def sym_quant_fpe3m1_real(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 6
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)
    w_sim=retain_first_mantissa_bit(w)
    w_sim = w_sim.mul(scales)
    w_sim = w_sim.reshape(w_shape)
    return w_sim



def get_scale(input, bits, mantissa_bit, bias):
        M = mantissa_bit
        E = bits - 1 - M
        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        minval = -maxval
        input = input.clamp(minval, maxval)
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)).detach(), 1.0)
        return input, 2.0 ** (input_log_scales - M - bias)

def sym_quant_fpe2m1(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 12
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)

    w, w_scale = get_scale(w,4,1,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)

    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape)
    return w_sim

# usage: w = sym_quant_fpe2m2(w,-1)
def sym_quant_fpe2m2(w, groupsize=-1):
    # fp8_scales = w.abs().max(dim=-1, keepdim=True)[0]
    # fp8_scales.div_(torch.finfo(torch.float8_e4m3fn).max)
    # w = w.div(fp8_scales).clamp(min=torch.finfo(torch.float8_e4m3fn).min, max=torch.finfo(torch.float8_e4m3fn).max).mul(fp8_scales)
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 14
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)

    w, w_scale = get_scale(w,5,2,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)

    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape)
    
    return w_sim

def sym_quant_fpe2m2_ps(w, groupsize=-1,power_scale=1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 1
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)
    sign = torch.sign(w)
    if isinstance(power_scale, int) or isinstance(power_scale, float):
        power_scale = torch.tensor(power_scale)
    else:
        power_scale=power_scale.reshape(-1,1)
    w = w.abs().pow(power_scale).mul(sign)
    w = w.mul(14)
    w, w_scale = get_scale(w,5,2,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)
    w_sim = w_sim.div(14)
    w_sign = torch.sign(w_sim)
    w_sim = w_sim.abs().pow(1/power_scale).mul(w_sign)
    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape).to(dtype=torch.float16)
    return w_sim



def sym_quant_fpe4m0(w, groupsize=-1):
    fp8_scales = w.abs().max(dim=-1, keepdim=True)[0]
    fp8_scales.div_(torch.finfo(torch.float8_e4m3fn).max)
    w = w.div(fp8_scales).clamp(min=torch.finfo(torch.float8_e4m3fn).min, max=torch.finfo(torch.float8_e4m3fn).max).mul(fp8_scales)
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 128
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)

    log_level = torch.round(torch.log2(w.abs()))
    sign = torch.sign(w)
    w = sign * 2 ** log_level
    w = torch.where(log_level < 0, 0, w)

    w = w.mul(scales)

    w = w.reshape(w_shape)
    return w


def sym_quant_fpe3m1(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 64*3
    scales.div_(q_max).clamp_(min=1e-5)
    w = w.div(scales)

    w, w_scale = get_scale(w,5,1,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)

    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape)
    return w_sim

def sym_quant_fpe1m3(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 15/4
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)

    w, w_scale = get_scale(w,5,3,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)

    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape)
    return w_sim


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_fpe2m2_fake(x,scale,power_scale=1):
    scale = scale.to(x.device)
    norm = x / scale
    # norm_pow = norm.abs().pow(power_scale).mul(torch.sign(norm))
    norm_scaled = norm.mul(14).clamp(-14,14)
    w, w_scale = get_scale(norm_scaled,5,2,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)
    w_sim = w_sim.div(14)
    # w_sim = norm_sim.abs().pow(1/power_scale).mul(torch.sign(norm_sim))
    w_sim = w_sim.mul(scale)
    return w_sim

def sym_dequant_fpe2m2(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

def asym_quant_int4(
    w, n_bit=4, q_group_size=-1
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    w = w.reshape(org_w_shape)
    return w


def int4_quant(
    w, n_bit=4, q_group_size=-1
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    max_val = w.abs().max(dim=1, keepdim=True)[0]
    max_val = max_val
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    scales = max_val / max_int
    scales.clamp_(min=1e-5)
    zeros = 0

    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales

    w = w.reshape(org_w_shape)

    return w

# usage: w = int5_quant(w, n_bit=5, q_group_size=-1)
def int5_quant(w, n_bit=5, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    max_val = w.abs().max(dim=1, keepdim=True)[0]
    max_val = max_val
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    scales = max_val / max_int
    scales.clamp_(min=1e-5)
    zeros = 0
    w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
    w = w.reshape(org_w_shape)
    return w


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.
def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0>=8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1>=8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)

class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
    def find_params_per_tensor(self, x):
        print('find_params_per_tensor')
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape
        reshaped_x = x.reshape(x.shape[0],-1)
        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            print(xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)
        cnt=1
        for shape in init_shape[1:]:
            cnt*=shape
        self.scale = self.scale.repeat(1, cnt).reshape(init_shape)
        self.zero = self.zero.repeat(1, cnt).reshape(init_shape)

class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, name=None, module:torch.nn.Linear=None):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.name = name
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.runtime_smooth = False
        self.out_runtime_smooth = False
        self.per_tensor = False
        self.act_scale_g128 = True
        self.scale_groupsize = 128

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype
        if self.quantizer.bits < 16: #Quantize, if needed
            if self.runtime_smooth:
                if len(x.shape) == 2:
                    act_scales = x.abs().max(dim=0,keepdim=True)[0]
                else:
                    act_scales = x.abs().max(dim=1,keepdim=True)[0]
                act_scales.clamp_(min=1e-5)
                if self.act_scale_g128:
                    index = torch.argsort(act_scales, dim=-1, descending=True)
                    act_scales = torch.gather(act_scales, -1, index)
                    sg = self.scale_groupsize
                    if len(x.shape) == 2:
                        act_scales = act_scales.reshape(1,x.shape[1]//sg,sg)
                        act_scales = act_scales.max(dim=-1,keepdim=True)[0].repeat(1,1,sg)
                        act_scales = act_scales.reshape(1,-1)
                    else:
                        act_scales = act_scales.reshape(x.shape[0],1,x.shape[2]//sg,sg)
                        act_scales = act_scales.max(dim=-1,keepdim=True)[0].repeat(1,1,1,sg)
                        act_scales = act_scales.reshape(x.shape[0],1,-1)
                    reverse_index = torch.argsort(index, dim=-1)
                    act_scales = torch.gather(act_scales, -1, reverse_index)
                x = x / act_scales
                x = x / 10

            q_max = torch.finfo(torch.float8_e4m3fn).max-10
            scale = x.abs().max(dim=-1, keepdim=True)[0]
            scale = scale.clamp(min=1e-5).div(q_max)
            x = x.div(scale).clamp(min=-q_max+1, max=q_max-1).to(torch.float8_e4m3fn).to(torch.float16).mul(scale)
            # x = x.clamp(min=-q_max+1, max=q_max-1).to(torch.float8_e4m3fn).to(torch.float16)
            if self.runtime_smooth:
                x = x * act_scales
                x = x * 10

            self.quantizer.free()

        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16: #Quantize the output, if needed
            if len(x.shape) == 2:
                act_scales = x.abs().max(dim=0,keepdim=True)[0]
            else:
                act_scales = x.abs().max(dim=1,keepdim=True)[0]
            act_scales.clamp_(min=1e-5)
            x = x / act_scales
            

            x = sym_quant_fpe2m2(x)

            x = x * act_scales

            self.out_quantizer.free()

        return x



class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8, 
        quant_func='int', power_scale=1
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.quant_func = quant_func
        self.power_scale = power_scale
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)
        if quant_func == 'fpe2m2':
            self.maxq = torch.tensor(1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    if self.quant_func == 'fpe2m2':
                        q = sym_quant_fpe2m2_fake(x, scale1.unsqueeze(1))
                    elif self.quant_func == 'int':
                        q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                    else:
                        raise RuntimeError('Unknown quantization function')
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.quant_func == 'fpe2m2':
                return sym_quant_fpe2m2_fake(x, self.scale,self.power_scale)
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)



def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(name=name + '.' + attr if name != '' else attr,module=tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(name=name + '.' + attr if name != '' else attr,module=child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(name=name + '.' + attr if name != '' else attr,module=child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)

def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
