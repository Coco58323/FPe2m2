# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/root/data/model/meta-llama/Llama-2-13b")
# %%
def retain_first_mantissa_bit(tensor):
    int_tensor = tensor.view(torch.int16)
    
    sign_bit = (int_tensor >> 15) & 1
    
    exponent_bits = (int_tensor >> 10) & 0x1F
    first_mantissa_bit = (int_tensor & 0x200)  
    second_mantissa_bit = (int_tensor & 0x100)
    first_and_sec = first_mantissa_bit & (second_mantissa_bit<<1)
    exponent_bits = torch.where(first_and_sec == 1, exponent_bits+1, exponent_bits)
    first_mantissa_bit = torch.where(first_and_sec == 1, 0, first_mantissa_bit)
    second_mantissa_bit = torch.where(first_and_sec == 1, 0, second_mantissa_bit)
    first_mantissa_bit = first_mantissa_bit | (second_mantissa_bit<<1)
    mask = torch.where(exponent_bits == 10, first_mantissa_bit, 1)
    exponent_bits = torch.where(mask == 0, 0, exponent_bits)
    exponent_bits = torch.where(exponent_bits < 10, 0, exponent_bits)
    first_mantissa_bit = torch.where(exponent_bits == 0, 0, first_mantissa_bit)
    new_int_tensor = (sign_bit << 15) | (exponent_bits << 10) | first_mantissa_bit
    new_tensor = new_int_tensor.view(torch.float16)

    return new_tensor

def sym_quant_fpe3m1_real(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 6
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)
    # w.clamp_(min=-4,max=4)
    w_sim=retain_first_mantissa_bit(w)
    w_sim = w_sim.mul(scales)
    w_sim = w_sim.reshape(w_shape)
    return w_sim


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
    fi_sec = torch.where(exponent_bits == 0, 0, fi_sec)
    first_mantissa_bit = fi_sec << 8
    new_int_tensor = (sign_bit << 15) | (exponent_bits << 10) | first_mantissa_bit
    new_tensor = new_int_tensor.view(torch.float16)
    return new_tensor

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

def retain_zero_mantissa_bit(tensor):
    int_tensor = tensor.view(torch.int16)
    sign_bit = (int_tensor >> 15) & 1
    exponent_bits = (int_tensor >> 10) & 0x1F

    first_mantissa_bit = (int_tensor >> 9) & 1
    exponent_bits = torch.where(first_mantissa_bit == 1, exponent_bits+1, exponent_bits)
    exponent_bits = torch.where(exponent_bits<9, 0, exponent_bits)

    print(exponent_bits.max())
    new_int_tensor = (sign_bit << 15) | (exponent_bits << 10)
    new_tensor = new_int_tensor.view(torch.float16)
    print(new_tensor.abs().max())
    return new_tensor

def sym_quant_fpe4m0_real(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 8
    scales.div_(q_max).clamp_(min=1e-5)
    w = w.div(scales)
    print(w.abs().max())
    w_sim=retain_zero_mantissa_bit(w)

    w_sim = w_sim.mul(scales)
    w_sim = w_sim.reshape(w_shape)
    return w_sim


def sym_quant_fpe2m2_real_mix(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 14
    scales.div_(q_max).clamp_(min=1e-5)
    w = w.div(scales)

    w_1, w_scale = get_scale(w,5,2,0)
    w_1=(w_1/w_scale).round()
    w_1=w_1.mul(w_scale)

    w_sim=retain_first_and_sec_mantissa_bit(w)

    w_sim = torch.where(w_1.abs()>=2, w_1, w_sim)
    w_sim = w_sim.mul(scales)
    w_sim = w_sim.reshape(w_shape)
    return w_sim

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

def sym_quant_fp4(w, groupsize=-1):
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

def sym_quant_fpe2m2(w, groupsize=-1):
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

def sym_quant_fpe3m1(w, groupsize=-1):
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 3*64
    scales.clamp_(min=1e-5).div_(q_max)
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
    q_max = 15/2
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)

    w, w_scale = get_scale(w,5,3,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)

    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape)
    return w_sim

def sym_quant_fpe2m2_ps(w, groupsize=-1,power_scale=0.5):
    fp8_scales = w.abs().max(dim=-1, keepdim=True)[0]
    fp8_scales.div_(torch.finfo(torch.float8_e4m3fn).max)
    w = w.div(fp8_scales).clamp(min=torch.finfo(torch.float8_e4m3fn).min, max=torch.finfo(torch.float8_e4m3fn).max).mul(fp8_scales)
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

def sym_quant_fpe3m1_ps(w, groupsize=-1,power_scale=1.2):
    fp8_scales = w.abs().max(dim=-1, keepdim=True)[0]
    fp8_scales.div_(torch.finfo(torch.float8_e4m3fn).max)
    w = w.div(fp8_scales).clamp(min=torch.finfo(torch.float8_e4m3fn).min, max=torch.finfo(torch.float8_e4m3fn).max).mul(fp8_scales)
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
    w = w.mul(6*13)
    w, w_scale = get_scale(w,5,1,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)
    w_sim = w_sim.div(6*13)
    w_sign = torch.sign(w_sim)
    w_sim = w_sim.abs().pow(1/power_scale).mul(w_sign)
    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape).to(dtype=torch.float16)
    return w_sim

# %%
import torch
matrix_1 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
matrix_2 = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.95]])

def count_frequency_matrix(matrix, bins=1000):
    min_val = 0.0
    max_val = 1.0
    bin_width = (max_val - min_val) / (bins/2)
    frequency_matrices = []
    for row in matrix:
        frequency_vector = torch.zeros(bins)
        bin_idx = ((row / bin_width)+(bins/2)).clamp(0, bins - 1).floor().int()
        for i in range(bins):
            frequency_vector[i] = (bin_idx == i).sum()
        frequency_matrices.append(frequency_vector)
    return torch.stack(frequency_matrices)


frequency_matrix_1 = count_frequency_matrix(matrix_1)
frequency_matrix_2 = count_frequency_matrix(matrix_2)

# 计算余弦相似度函数
def cosine_similarity_matrix(freq_matrix_1, freq_matrix_2):
    dot_products = (freq_matrix_1 * freq_matrix_2).sum(dim = 1)
    magnitudes_1 = freq_matrix_1.norm(dim = 1)
    magnitudes_2 = freq_matrix_2.norm(dim = 1)
    average_score = dot_products / (magnitudes_1 * magnitudes_2)
    average_score = average_score.mean()
    return average_score


print("按行的余弦相似度:", cosine_similarity_matrix(frequency_matrix_1, frequency_matrix_2))
# %%
cnt=1
import matplotlib.pyplot as plt
for name, param in model.named_parameters():
    if "lm_head" in name or "layer" not in name or 'norm' in name:
        continue
    param = param.to(torch.float16).detach()[:1,:]
    scale = param.abs().max()
    param = param.div(scale)
    fpe3m1 = sym_quant_fpe3m1(param)
    fpe3m1_ps = sym_quant_fpe3m1_ps(param)
    fpe2m2 = sym_quant_fpe2m2(param)
    fpe2m2_ps = sym_quant_fpe2m2_ps(param)
    fpe1m3 = sym_quant_fpe1m3(param)
    
    # cnt_param = len(set(param[0].detach().cpu().numpy().tolist()))
    # cnt_fpe3m1 = len(set(fpe3m1[0].detach().cpu().numpy().tolist()))
    # cnt_fpe2m2 = len(set(fpe2m2[0].detach().cpu().numpy().tolist()))
    # cnt_fpe1m3 = len(set(fpe1m3[0].detach().cpu().numpy().tolist()))
    # print(cnt_param, cnt_fpe3m1)
    # print(cnt_fpe2m2, cnt_fpe1m3)
    pa = param[0].detach()
    pa = pa / pa.abs().max()
    plt.hist(pa.cpu().numpy(), bins=5000)
    plt.show()
    pa = pa.abs().pow(0.5).mul(torch.sign(pa))
    pa = pa.cpu().numpy()
    plt.hist(pa, bins=5000)
    plt.show()
    # plt.hist(fpe3m1[0].detach().cpu().numpy(), bins=100,color='b')
    # plt.hist(fpe3m1_ps[0].detach().cpu().numpy(), bins=100,color='r',alpha=0.5)
    # plt.show()
    plt.hist(fpe2m2[0].detach().cpu().numpy(), bins=100,color='b')
    plt.hist(fpe2m2_ps[0].detach().cpu().numpy(), bins=100,color='r',alpha=0.5)
    plt.show()
    plt.hist(fpe1m3[0].detach().cpu().numpy(), bins=100)
    plt.show()
    # fpe3m1 = sym_quant_fpe3m1(param)
    # int5 = int4_quant(param, 5, -1)
    freq_origin = count_frequency_matrix(param)
    freq_fpe3m1 = count_frequency_matrix(fpe3m1)
    freq_fpe2m2 = count_frequency_matrix(fpe2m2)
    freq_fpe1m3 = count_frequency_matrix(fpe1m3)
    # freq_fpe3m1 = count_frequency_matrix(fpe3m1)
    # freq_int5 = count_frequency_matrix(int5)
    # plt.bar(range(1000), freq_origin[0].detach().cpu().numpy())
    # plt.show()
    # plt.bar(range(1000), freq_fpe3m1[0].detach().cpu().numpy())
    # plt.show()
    # print(name)
    # print("origin and fpe3m1:", cosine_similarity_matrix(freq_origin, freq_fpe3m1))
    # print("origin and fpe2m2:", cosine_similarity_matrix(freq_origin, freq_fpe2m2))
    # print("origin and fpe1m3:", cosine_similarity_matrix(freq_origin, freq_fpe1m3))
    # print("origin and fpe3m1:", cosine_similarity_matrix(freq_origin, freq_fpe3m1))
    # print("origin and int5:", cosine_similarity_matrix(freq_origin, freq_int5))
    # break
    cnt -= 1
    if cnt<0:
        break
# %%
def sym_quant_fpe2m2_roll3(w, groupsize=-1,pow_scale=1):
    fp8_scales = w.abs().max(dim=-1, keepdim=True)[0]
    fp8_scales.div_(torch.finfo(torch.float8_e4m3fn).max)
    w = w.div(fp8_scales).clamp(min=torch.finfo(torch.float8_e4m3fn).min, max=torch.finfo(torch.float8_e4m3fn).max).mul(fp8_scales)
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 1
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)
    sign = torch.sign(w)
    w = w.abs().pow(pow_scale).mul(sign)
    w = w.mul(14)
    w, w_scale = get_scale(w,5,2,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)
    w_sim = w_sim.div(14)
    w_sign = torch.sign(w_sim)
    w_sim = w_sim.abs().pow(1/pow_scale)
    w_sim = w_sim.mul(w_sign)
    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape)
    return w_sim
# %%
import tqdm
cnt=40
for name, param in model.named_parameters():
    if "lm_head" in name or "layer" not in name or 'norm' in name:
        continue
    # pring width
    torch.set_printoptions(linewidth=300)
    param = param.to(torch.float16).detach()[:10,:]
    # norm the weight
    scale = param.abs().max(dim=-1, keepdim=True)[0]
    normed_param = param.div(scale)
    mean = normed_param.mean(dim=-1)
    varians = normed_param.var(dim=-1)
    recon_loss = []
    for i in range(50, 121):
        pow_scale = i/100
        w = sym_quant_fpe2m2_roll3(param, pow_scale=pow_scale)
        loss = (param - w).float().pow(2).mean(dim=-1)
        # print(f"pow_scale: {pow_scale}")
        # print(f"Loss: {loss}")
        recon_loss.append(loss)
    recon_loss = torch.stack(recon_loss)
    # get the best pow_scale for each weight and its index
    best_pow_scale, best_index = recon_loss.min(dim=0)
    best_index = (best_index+50)/100
    # print(f"Best pow_scale: {best_pow_scale}")
    print(name)
    print(f"Best index: {best_index}")
    print(best_index.min(), best_index.max())
    cnt -= 1
    if cnt<0:
        break
# %%
# analyze the relationship between the variance and the best index
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(5000)
y1 = best_index.cpu().numpy()
y2 = varians.cpu().numpy()
plt.plot(x, y1, label="Best index")
plt.plot(x, y2, label="Variance")

# %%
# y = 59.2900 + 1111.8118x
y3 = 0.592900 + 11.118118 * y2
y1_real = y1 / 100
x = np.arange(5000)
plt.plot(x, y1_real, label="Best index")
plt.plot(x, y3, label="Linear regression")
# %%
import numpy as np
from sklearn.linear_model import LinearRegression

# 假设两行数据，第一行为自变量x，第二行为因变量y
x_data = y2.reshape(-1, 1)
y_data = y1.reshape(-1, 1)

# 创建线性回归模型对象
model = LinearRegression()

# 使用数据拟合模型
model.fit(x_data, y_data)

# 输出回归系数（斜率）和截距
print("斜率:", model.coef_[0][0])
print("截距:", model.intercept_[0])

# 可以使用模型进行预测
new_x = np.array([[6]])
predicted_y = model.predict(new_x)
print("预测值:", predicted_y[0][0])
# %%
import numpy as np
from scipy.stats import pearsonr

# 假设两行数据，第一行为自变量x，第二行为因变量y
x_data = y2
y_data = y1

# 使用pearsonr函数计算Pearson相关性系数和p - 值
corr_coef, p_value = pearsonr(x_data, y_data)

print("Pearson相关性系数:", corr_coef)
print("p - 值:", p_value)
# %%
import numpy as np
from scipy.optimize import curve_fit

# 假设的二次函数数据（示例数据）
x_data = y2
y_data = y1/100

# 定义二次函数模型
def quadratic_function(x, a, b, c, d, e):
    return a*x**(1/2) + b*x + c + d*x**2 + e*x**(1/3)

# 使用curve_fit进行拟合
popt, pcov = curve_fit(quadratic_function, x_data, y_data)

# 输出拟合得到的系数
print("二次函数拟合系数: a =", popt[0], "b =", popt[1], "c =", popt[2], "d =", popt[3], "e =", popt[4])
# %%
y3 = quadratic_function(y2, popt[0], popt[1], popt[2], popt[3], popt[4])
x = np.arange(5000)
plt.plot(x, y1/100, label="Best index")
plt.plot(x, y3, label="Quadratic regression")
# %%
# a = 0.7931087222168258 b = 2.215173024571522 c = 0.40562785196611995 d = -42.116633767648054 e = 1.0458278433898283
def sym_quant_fpe2m2_autoscale(w, groupsize=-1):
    fp8_scales = w.abs().max(dim=-1, keepdim=True)[0]
    fp8_scales.div_(torch.finfo(torch.float8_e4m3fn).max)
    w = w.div(fp8_scales).clamp(min=torch.finfo(torch.float8_e4m3fn).min, max=torch.finfo(torch.float8_e4m3fn).max).mul(fp8_scales)
    w_shape = w.shape
    if groupsize > 0:
        assert w_shape[-1] % groupsize == 0
        w = w.reshape(w_shape[0], -1, groupsize)

    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 1
    scales.clamp_(min=1e-5).div_(q_max)
    w = w.div(scales)
    sign = torch.sign(w)
    variances = w.var(dim=-1,keepdim=True)
    pow_scale = quadratic_function(variances, 0.7931087222168258, 2.215173024571522, 0.40562785196611995, -42.116633767648054, 1.0458278433898283)
    pow_scale = pow_scale.clamp(max=1)
    w = w.abs().pow(pow_scale).mul(sign)
    w = w.mul(14)
    w, w_scale = get_scale(w,5,2,0)
    w=(w/w_scale).round()
    w_sim=w.mul(w_scale)
    w_sim = w_sim.div(14)
    w_sign = torch.sign(w_sim)
    w_sim = w_sim.abs().pow(1/pow_scale)
    w_sim = w_sim.mul(w_sign)
    w_sim = w_sim.mul(scales)

    w_sim = w_sim.reshape(w_shape)
    return w_sim
# %%
cnt = 10
for name, param in model.named_parameters():
    if "lm_head" in name or "layer" not in name or 'norm' in name:
        continue
    print(name)
    torch.set_printoptions(linewidth=300)
    param = param.to(torch.float16).detach()
    w1 = sym_quant_fpe2m2_autoscale(param)
    w2 = sym_quant_fpe2m2(param)
    loss1 = (param - w1).float().pow(2).mean()
    loss2 = (param - w2).float().pow(2).mean()
    # compare the loss
    print(loss1.mean())
    print(loss2.mean())
    cnt -= 1
    if cnt<0:
        break

# %%
for name, param in model.named_parameters():
    if "lm_head" in name or "layer" not in name or 'norm' in name:
        continue
    param = param.to(torch.float16).detach()[:1,:]
    scale = param.abs().max()/14
    param = param.div(scale)
    norm = param.div(14)
    sign = torch.sign(norm)
    norm = norm.abs().pow(0.3).mul(sign)
    norm = norm.mul(14)
    print(norm)
    # set figure size
    plt.figure(figsize=(20, 5))
    plt.hist(param[0].detach().cpu().numpy(), bins=5000)
    plt.show()
    plt.figure(figsize=(20, 5))
    plt.hist(norm[0].detach().cpu().numpy(), bins=5000)
    break
# %%
import torch
import numpy as np
e2m2 = torch.tensor([0,1/2,1,3/2,2,5/2,3,7/2,4,5,6,7,8,10,12,14]) 
e3m1 = torch.tensor([0,1/16,2/16,3/16,1/4,3/8,1/2,3/4,1,3/2,2,3,4,6,8,12]) * 14/12
int5 = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) * 14/15
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
e2m2 = e2m2.numpy()
e3m1 = e3m1.numpy()
int5 = int5.numpy()
plt.plot(e2m2, [1]*len(e2m2), 'ro')
plt.plot(e3m1, [2]*len(e3m1), 'bo')
plt.plot(int5, [3]*len(int5), 'go')
# not plot y axis
plt.yticks([])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 均值和标准差
mu = 0
sigma = 1

# 在均值附近生成数据点
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
# 计算高斯分布的概率密度函数值
y = norm.pdf(x, mu, sigma)

plt.plot(x, y)
plt.title('Gaussian Distribution (Normal Distribution)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.show()

# %%
