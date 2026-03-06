import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os

# 设置基础路径
BASE_DIR = "/workspace/Qwen-1.7B"  # 你的工作目录

# 切换到正确目录
os.chdir(BASE_DIR)
print(f"当前工作目录: {os.getcwd()}")
print("目录中的文件:")
for file in os.listdir('.'):
    print(f"  - {file}")

os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []
    
    # 使用绝对路径
    origin_full_path = os.path.join(BASE_DIR, origin_path)
    new_full_path = os.path.join(BASE_DIR, new_path)
    
    print(f"读取文件: {origin_full_path}")
    
    # 检查源文件是否存在
    if not os.path.exists(origin_full_path):
        print(f"错误：源文件不存在 {origin_full_path}")
        return False

    # 读取旧的JSONL文件
    with open(origin_full_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            output = f"<think>{data['think']}</think> \n {data['answer']}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_full_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    print(f"成功转换 {len(messages)} 条数据到 {new_full_path}")
    return True


    """验证数据集是否有问题"""
    print("验证数据集...")
    problematic_indices = []
    
    for idx, row in df.iterrows():
        # 检查空值
        if pd.isna(row['input']) or pd.isna(row['output']):
            print(f"⚠️ 样本 {idx} 存在空值")
            problematic_indices.append(idx)
            continue
            
        # 检查长度
        if len(row['input']) == 0 or len(row['output']) == 0:
            print(f"⚠️ 样本 {idx} 内容为空")
            problematic_indices.append(idx)
            continue
            
        # 检查异常字符
        if '<think>' not in row['output']:
            print(f"⚠️ 样本 {idx} 缺少<think>标签")
            # 不一定是问题，只是提醒
            
    if problematic_indices:
        print(f"发现 {len(problematic_indices)} 个问题样本，将被移除")
        df = df.drop(problematic_indices).reset_index(drop=True)
    
    print(f"验证完成，剩余 {len(df)} 个有效样本")
    return df

def process_func(example):
    """
    将数据集进行预处理
    """ 
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

'''
下载模型到指定目录
print("开始下载模型...")
model_dir = snapshot_download(
    "Qwen/Qwen3-1.7B", 
    cache_dir=os.path.join(BASE_DIR, "model_cache"),  # 模型缓存目录
    revision="master"
)
print(f"模型下载到: {model_dir}")
'''

# 使用已下载的模型 - 关键修改！
model_path = os.path.join(BASE_DIR, "Qwen/Qwen3-1.7B")  # 你已下载的模型路径

print(f"使用已下载的模型: {model_path}")

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"错误：找不到模型文件 {model_path}")
    print("请确认模型已下载到正确位置")
    exit(1)

# 列出模型目录中的文件
print("模型目录中的文件:")
for file in os.listdir(model_path):
    print(f"  - {file}")

# 加载模型
print("加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    use_fast=False, 
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)

model.enable_input_require_grads()

# 9. 配置LoRA
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 设置数据文件路径
train_dataset_path = "train.jsonl"  # 你的训练数据文件
test_dataset_path = "val.jsonl"      # 你的验证数据文件

train_jsonl_new_path = "train_format.jsonl"  # 转换后的训练数据
test_jsonl_new_path = "val_format.jsonl"      # 转换后的验证数据

# 检查数据文件是否存在
print("\n检查数据文件...")
train_file = os.path.join(BASE_DIR, train_dataset_path)
test_file = os.path.join(BASE_DIR, test_dataset_path)

if os.path.exists(train_file):
    print(f"✅ 找到训练文件: {train_file}")
    file_size = os.path.getsize(train_file) / 1024  # KB
    print(f"   文件大小: {file_size:.2f} KB")
else:
    print(f"❌ 找不到训练文件: {train_file}")
    exit(1)

if os.path.exists(test_file):
    print(f"✅ 找到验证文件: {test_file}")
    file_size = os.path.getsize(test_file) / 1024
    print(f"   文件大小: {file_size:.2f} KB")
else:
    print(f"❌ 找不到验证文件: {test_file}")
    test_dataset_path = None

# 转换数据格式
print("\n转换数据格式...")
if not os.path.exists(os.path.join(BASE_DIR, train_jsonl_new_path)):
    success = dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
    if not success:
        print("数据转换失败，退出程序")
        exit(1)
else:
    print(f"训练格式化文件已存在: {train_jsonl_new_path}")

if test_dataset_path and not os.path.exists(os.path.join(BASE_DIR, test_jsonl_new_path)):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 加载训练集
print("\n加载训练集...")
train_df = pd.read_json(os.path.join(BASE_DIR, train_jsonl_new_path), lines=True)
print(f"训练集大小: {len(train_df)} 条")
print("训练集样例:")
print(train_df.head(1))
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 加载验证集
if test_dataset_path and os.path.exists(os.path.join(BASE_DIR, test_jsonl_new_path)):
    print("加载验证集...")
    eval_df = pd.read_json(os.path.join(BASE_DIR, test_jsonl_new_path), lines=True)
    print(f"验证集大小: {len(eval_df)} 条")
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)
else:
    print("从训练集分割10%作为验证集")
    train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-1.7B",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    run_name="qwen3-1.7B",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

print("开始训练...")
trainer.train()

print("训练完成！保存模型...")
trainer.save_model(os.path.join(BASE_DIR, "final_model"))


    