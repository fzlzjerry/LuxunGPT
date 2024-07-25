import random
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import Dataset
import pickle
import os

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 加载模型和分词器
model_name = 'llama3.1-3b'  # 替换成Llama 3.1: 8B模型的名称
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 加载二进制数据
def load_bin_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return Dataset.from_dict(data)

train_data_path = '/mnt/data/train.bin'
val_data_path = '/mnt/data/val.bin'

if os.path.exists(train_data_path) and os.path.exists(val_data_path):
    train_data = load_bin_data(train_data_path)
    val_data = load_bin_data(val_data_path)
else:
    raise FileNotFoundError("训练或验证数据文件不存在。")

# 数据整理器
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 自定义早停回调
class MyEarlyStoppingCallback(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step >= args.max_steps:
            control.should_training_stop = True
            print("早停：验证损失未改善，训练停止。")

# 训练参数
train_args = TrainingArguments(
    output_dir = './results',  # 输出目录
    overwrite_output_dir = True,  # 覆盖输出目录
    num_train_epochs = 200,  # 训练周期数
    per_device_train_batch_size = 2,  # 每设备训练批次大小
    save_steps = 1000,  # 每1000步保存一次
    save_total_limit = 2,  # 最大保存模型数
    logging_dir = './logs',  # 日志目录
    logging_steps = 200,  # 每200步记录一次日志
    fp16 = True,  # 启用混合精度训练
    evaluation_strategy = "steps",  # 训练期间评估策略
    eval_steps = 200,  # 评估步数
    report_to="tensorboard",  # 使用tensorboard记录
    gradient_accumulation_steps = 4,  # 梯度累积步数
    dataloader_num_workers = 4,  # 数据加载子进程数
    learning_rate = 5e-5,  # 学习率
    weight_decay = 0.01,  # 权重衰减
    warmup_steps = 500,  # 预热步数
    max_grad_norm = 1.0,  # 梯度裁剪
    logging_first_step = True,  # 记录第一个步数
    no_cuda = False,  # 使用GPU
    seed = 42,  # 随机种子
    local_rank = -1,  # 本地进程号
    disable_tqdm = False,  # 启用tqdm进度条
    remove_unused_columns = True,  # 移除未使用的列
    label_names = ["labels"],  # 标签名称
    load_best_model_at_end = True,  # 在结束时加载最佳模型
    metric_for_best_model = "eval_loss",  # 最佳模型评估指标
    lr_scheduler_type = "linear",  # 学习率调度器类型
    warmup_ratio = 0.1,  # 预热比例
)

# 多GPU并行训练
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# 训练器
trainer = Trainer(
    model = model,  # 模型
    args = train_args,  # 训练参数
    data_collator = collator,  # 数据整理器
    train_dataset = train_data,  # 训练数据集
    eval_dataset = val_data,  # 验证数据集
    callbacks = [MyEarlyStoppingCallback(early_stopping_patience = 5)]  # 自定义早停回调
)

# 训练模型
try:
    trainer.train()
except Exception as e:
    print(f"训练错误: {e}")

# 保存模型
model.module.save_pretrained('./model') if torch.cuda.device_count() > 1 else model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

print("训练和保存成功。")