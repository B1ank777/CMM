10. 用 MemTorch 映射 Transformer 的 Linear 层

MemTorch 的 patch_model 可以把 PyTorch 模型里的指定模块转换成 memristive layer；官方文档说明它支持对 torch.nn.Linear、torch.nn.Conv2d 等模块进行 patch。

第一版建议只 patch decoder 和输出层里的 nn.Linear，不要 patch CNN encoder。因为 CNN encoder 很大，调试成本高。等 decoder 跑通之后，再考虑 patch VGG/ResNet 的线性或卷积层。

10.1 先冻结 encoder
for p in model.encoder.parameters():
    p.requires_grad = False
10.2 转换整个模型中的 Linear
import copy
import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.map.Input import naive_scale
from memtorch.bh.crossbar.Program import naive_program

def build_memristive_model(model):
    mem_model = patch_model(
        copy.deepcopy(model),
        memristor_model=memtorch.bh.memristor.VTEAM,
        memristor_model_params={
            "r_on": 1e2,
            "r_off": 1e4
        },
        module_parameters_to_patch=[torch.nn.Linear],
        mapping_routine=naive_map,
        transistor=True,
        programming_routine=naive_program,
        tile_shape=(128, 128),
        max_input_voltage=0.3,
        scaling_routine=naive_scale,
        ADC_resolution=8,
        ADC_overflow_rate=0.0,
        quant_method="linear",
        use_bindings=True
    )

    return mem_model

然后：

model.load_state_dict(torch.load("caption_transformer_epoch_9.pt", map_location=device))
model.eval()

mem_model = build_memristive_model(model)
mem_model.to(device)
mem_model.eval()

如果 use_bindings=True 报错，改成：

use_bindings=False

如果 VTEAM 参数不匹配你的 MemTorch 版本，先用空参数：

memristor_model_params={}

因为不同版本的 MemTorch 对 device model 参数要求可能略有差异。MemTorch 文档中也给出了使用 VTEAM、Crossbar、naive_map、naive_scale 等接口的典型示例。