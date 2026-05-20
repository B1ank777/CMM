12. 对应原论文的 it-8、it-9、it-10 怎么模拟？

原论文的 it-10、it-9、it-8 本质上对应不同写入脉冲条件 i?dt，影响权重写入误差。Table 2 显示，it-8 的 BLEU-1 为 0.629，参考网络为 0.646，达到约 97.37% 的参考性能；作者因此推荐 it-8 作为硬件实现折中方案。

MemTorch 不一定直接叫 “it-8”，所以你可以用两层方式模拟：

第一层：MemTorch 自带非理想性

MemTorch 提供 nonideality 模块，可以向 patched model 引入器件和电路非理想性。官方文档说明 apply_nonidealities 可用于引入各种 device/circuit characteristics。

你可以先做：

from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
from memtorch.bh.nonideality import NonIdeality

mem_model = mem_model.apply_nonidealities(
    mem_model,
    non_idealities=[
        NonIdeality.FiniteConductanceStates,
        NonIdeality.DeviceFaults
    ],
    conductance_states=256,
    lrs_proportion=0.01,
    hrs_proportion=0.01
)

不同 MemTorch 版本的参数名可能会有差异，所以这里要以你本地 help(mem_model.apply_nonidealities) 为准。

第二层：自己注入写入误差

为了贴近原论文，可以在训练好的 PyTorch 权重上手动加噪声：

def inject_write_noise(model, noise_std):
    noisy_model = copy.deepcopy(model)

    with torch.no_grad():
        for name, param in noisy_model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)

    return noisy_model

你可以定义：

noise_settings = {
    "it-10": 1e-6,
    "it-9": 1e-5,
    "it-8": 1e-4,
    "it-7": 1e-3,
    "it-6": 1e-2,
}

然后每个条件都评估一次：

for condition, noise_std in noise_settings.items():
    noisy_model = inject_write_noise(model, noise_std)
    mem_model = build_memristive_model(noisy_model)
    scores = evaluate_model(mem_model, val_loader, vocab, device)
    print(condition, scores)

注意：这些 noise_std 不是严格等价于原论文的 i?dt，但可以作为第一版实验近似。更严谨的做法是根据原论文公式把 i?dt 转成每层权重写入误差上界，再注入 layer-wise noise。