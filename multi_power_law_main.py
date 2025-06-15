import os
os.chdir(os.path.dirname(__file__))

# 使用 Multi-Power Law 模型在一个或两个 schedule 上拟合，并预测其他 schedule

import logging
logging.getLogger().addHandler(logging.NullHandler())

import numpy as np
import pandas as pd

from src.fitting import generate_init_params, initialize_params, mpl_adam_fit
from src.models import MPL
from src.evaluation import evaluate_mpl

# ---------------------------- 配置部分 ----------------------------
pkl_path = "./loss_curve_repo/loss_curves/gpt_loss+lrs.pkl"
# keys = ['M:100M_gpt_D:20B_scheduler:811_rope',
#         'M:100M_gpt_D:20B_scheduler:wsd_rope',
#         'M:100M_gpt_D:20B_scheduler:cosine_rope'
# ]

# train_keys = [
#     'M:100M_gpt_D:20B_scheduler:811_rope',
#     'M:100M_gpt_D:20B_scheduler:cosine_rope',
# ]  # 拟合时使用的 schedule key

train_keys = [
    '811',
    # 'wsd',
    'cosine',
] # 拟合时使用的 schedule key

# 输出路径
foldername = '_'.join(train_keys) # 按 train_keys 确定输出到哪个目录
output_dir = './fit/' + foldername
os.makedirs(output_dir, exist_ok=True)

# multi-power law 需要一个 warmup
WARMUP_OFFSET = 2160

# ---------------------------- 加载数据 ----------------------------

# 读取数据
data_raw = pd.read_pickle(pkl_path)

# 把 key 按 LRS 截断，以免包含 : 等非法符号，生成文件时不方便
data_raw = {k[27:-5]: v for k, v in data_raw.items()}

# 自动推断测试集 key
test_keys = [k for k in data_raw.keys() if k not in train_keys]

# ---------------------------- 数据标准化 ----------------------------
data_dict = {}
for key, df in data_raw.items():
    df = df.sort_values("step").reset_index(drop=True)

    step = df["step"].astype(int).values
    loss = df["Metrics/loss"].astype(float).values

    # lrs  = df["lr"].astype(float).values

    # 给 lrs 插值，因为 step 中有缺失的位置
    step_min = df["step"].min()
    step_max = df["step"].max()
    full_step = np.arange(step_min, step_max+1)
    df_interp = df.set_index("step")
    lrs = df_interp["lr"].reindex(full_step).interpolate(method='linear').astype(float).values

    # 我们的数据不包含 warmup，所以要手动构造一段
    step = step + WARMUP_OFFSET
    lrs = np.concatenate([np.linspace(0, lrs[0], num=WARMUP_OFFSET, endpoint=False), lrs])

    # !! 我们不能这么做，因为811等在24000步以后才衰减
    # data_loader 这样做，但可能没有必要
    # if step_max > 24000:
    #     mask = step < 24000
    #     step = step[mask]
    #     loss = loss[mask]

    mask = step % 10 == 0
    step = step[mask]
    loss = loss[mask]

    data_dict[key] = {"step": step, "loss": loss, "lrs": lrs}

# ---------------------------- 拟合模型 ----------------------------
init_param = initialize_params(data_dict, train_keys)
init_params = generate_init_params(init_param)
best_params, best_loss = mpl_adam_fit(data_dict, train_keys, test_keys, init_params, output_dir)

# Evaluate
evaluate_mpl(data_dict, train_keys, best_params, output_dir, 'Training set: '+foldername)
evaluate_mpl(data_dict, test_keys, best_params, output_dir, 'Training set: '+foldername)
# logger.info(f"Best Loss: {best_loss}")
print(f"Best Loss: {best_loss}")

print("\n所有拟合与预测完成。")
