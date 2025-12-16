import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 加载预训练的模型和分词器，设置二分类任务（num_labels=2）
model_name = "albert/albert-base-v1"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 加载 IMDb 数据集（此处使用 "stanfordnlp/imdb"，如果数据集名称不一致，也可以尝试 "imdb"）
dataset = load_dataset("stanfordnlp/imdb", trust_remote_code=True)

# 3. 定义预处理函数，对文本进行分词、填充和截断
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# 对数据集进行批量预处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. 获取训练集和验证集（这里数据集默认包含 train 和 test 分割）
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",              # 输出目录
    evaluation_strategy="epoch",         # 每个 epoch 后评估一次
    learning_rate=1e-5,                  # 学习率
    per_device_train_batch_size=8,       # 训练时每个设备的 batch size
    per_device_eval_batch_size=8,        # 验证时每个设备的 batch size
    num_train_epochs=3,                  # 训练轮数
    weight_decay=0.01,                   # 权重衰减
    logging_dir="./logs",                # 日志目录
    logging_steps=10,                    # 每 10 步记录一次日志
    save_strategy="epoch",               # 每个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载表现最好的模型
)

# 6. 初始化 Trainer，并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

# 7. 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 8. 保存微调后的模型和分词器
model.save_pretrained("./fine-tuned-albert")
tokenizer.save_pretrained("./fine-tuned-albert")
