import argparse
import os
import sys
import time
import torch
import logging
import pandas as pd

from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

from peft import (
    get_peft_model,
    PeftType,
    PrefixTuningConfig,
    PeftModel
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)

from tqdm import tqdm

# Argument Parsing
parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_name_or_path", type=str, default="./LLMs/graphcodebert-base")
parser.add_argument("--train_file", type=str, default="dataset/牛逼/train.csv")
parser.add_argument("--valid_file", type=str, default="dataset/牛逼/valid.csv")
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_train_samples", type=int, choices=[100, 200, 500, 1000], default=None)
parser.add_argument("--num_virtual_tokens", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--optimizer", type=str, default="Adamw")
parser.add_argument("--should_log", type=bool, default=True)
parser.add_argument("--output_dir", type=str, default="output")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

model_name = args.model_name_or_path.split("/")[-1]

# Setup logging
logger = logging.getLogger(__name__)
log_file_path = os.path.join(args.output_dir, model_name, f"prefix_tuning_seed_{args.seed}", f"train_max_samples_{args.max_train_samples}_log.txt")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)

logging.getLogger().handlers[0].setLevel(logging.WARNING)
logger.setLevel(logging.INFO if args.should_log else logging.WARN)

content_write = "=" * 50 + "\n"
content_write += "Prefix Tuning\n"
content_write += f"seed: {args.seed}\n"
content_write += f"model_name_or_path: {args.model_name_or_path}\n"
content_write += f"train_file: {args.train_file}\n"
content_write += f"valid_file: {args.valid_file}\n"
content_write += f"max_seq_length: {args.max_seq_length}\n"
content_write += f"batch_size: {args.batch_size}\n"
content_write += f"num_epochs: {args.num_epochs}\n"
content_write += f"max_train_samples: {args.max_train_samples}\n"
content_write += f"num_virtual_tokens: {args.num_virtual_tokens}\n"
content_write += f"learning_rate {args.learning_rate:.0e}\n"
content_write += f"optimizer: {args.optimizer}\n"
content_write += f"should_log: {args.should_log}\n"
content_write += f"output_dir: {args.output_dir}\n"
content_write += "=" * 50 + "\n"
print(content_write)
logger.info(content_write)

set_seed(args.seed)
use_cuda = True

# Set peft config
peft_config = PrefixTuningConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=args.num_virtual_tokens
)

if "codet5" in args.model_name_or_path.lower():
    logging.warning("CodeT5 models are not supported yet. Please use another model.")
    sys.exit(1)

# Load tokenizer
padding_side = PADDING_SIDE.get(model_name, "right")  # 防御性处理

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

# === ✅ 新增：定义 clean_text 函数（与 LoRA 脚本一致）===
def clean_text(text):
    """清理单个文本字符串"""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = text.replace("\n", " ")
    return text

# Load datasets
train_df = pd.read_csv(args.train_file, encoding='ISO-8859-1')
eval_df = pd.read_csv(args.valid_file, encoding='ISO-8859-1')

# === ✅ 关键修复 1: 清洗 label 为整数（与 LoRA 一致）===
def ensure_int_labels(df, split_name):
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    original_len = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    if len(df) < original_len:
        logger.warning(f"Dropped {original_len - len(df)} rows in {split_name} due to invalid labels.")
    df["label"] = df["label"].astype(int)
    return df

train_df = ensure_int_labels(train_df, "train")
eval_df = ensure_int_labels(eval_df, "valid")

# 如果指定了 max_train_samples，则采样
if args.max_train_samples is not None:
    train_df = train_df.sample(n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)

trainset = Dataset.from_pandas(train_df)
evalset = Dataset.from_pandas(eval_df)

datasets = DatasetDict({
    'train': trainset,
    'validation': evalset
})

# Get number of labels
num_labels = len(train_df["label"].unique())
logger.info(f"Number of labels: {num_labels}")

# Tokenize datasets
if args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({args.max_seq_length}) is larger than the model's maximum length "
        f"({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

# === ✅ 修复：条件判断错误（原代码 "bert" or "unixcoder" 永远为 True）===
model_path_lower = args.model_name_or_path.lower()
if "bert" in model_path_lower or "unixcoder" in model_path_lower:
    tokenize_max_length = max_seq_length - args.num_virtual_tokens
else:
    tokenize_max_length = max_seq_length

def tokenize_function(examples):
    # === ✅ 应用 clean_text（与 LoRA 一致）===
    cleaned_texts = [clean_text(t) for t in examples["text"]]
    outputs = tokenizer(
        cleaned_texts,
        truncation=True,
        padding="max_length",
        max_length=tokenize_max_length
    )
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "text_label"] if "text_label" in datasets["train"].column_names else ["text"],
    load_from_cache_file=False
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    # === ✅ 更安全的 collate：显式处理 labels（可选但推荐）===
    labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
    input_keys = ["input_ids", "attention_mask"]
    inputs = {k: [ex[k] for ex in examples] for k in input_keys}
    padded = tokenizer.pad(inputs, return_tensors="pt", padding="max_length", max_length=max_seq_length)
    padded["labels"] = labels
    return padded

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
logger.info(f"Prefix Tuning - Trainable parameters: {model.get_nb_trainable_parameters()}")

# === ✅ 修复：条件判断错误（原代码永远为 True）===
if "deepseekcoder" in model_path_lower or "starcoder" in model_path_lower:
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

# Optimizer & Scheduler
if args.optimizer.lower() == "adamw":
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
    num_training_steps=(len(train_dataloader) * args.num_epochs)
)

total_steps = 0
best_validation_loss = float("inf")
peak_memory = 0
if use_cuda:
    model.cuda()

# Training loop
start_time = time.time()
for epoch in range(args.num_epochs):
    model.train()
    train_loss = 0.0

    progress_bar_train = tqdm(
        total=len(train_dataloader),
        desc=f"Training epoch {epoch + 1}",
        position=0,
        mininterval=1,
        leave=True
    )

    for step, batch in enumerate(train_dataloader):
        total_steps += 1
        batch = {k: v.cuda() if use_cuda else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 5 == 0:
            progress_bar_train.set_postfix({"loss": loss.item()})
            progress_bar_train.update(5)

        current_memory = torch.cuda.max_memory_allocated()
        if current_memory > peak_memory:
            peak_memory = current_memory

    progress_bar_train.close()
    avg_train_loss = train_loss / len(train_dataloader)
    logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss}")
    print(f"Epoch {epoch + 1} - Training loss: {avg_train_loss}")

    # Validation
    model.eval()
    total_validation_loss = 0.0
    progress_bar_valid = tqdm(
        total=len(valid_dataloader),
        desc=f"Validation epoch {epoch + 1}",
        position=0,
        mininterval=1,
        leave=True
    )

    for step, batch in enumerate(valid_dataloader):
        batch = {k: v.cuda() if use_cuda else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_validation_loss += loss.item()
        if step % 5 == 0:
            progress_bar_valid.update(5)
    progress_bar_valid.close()

    avg_validation_loss = total_validation_loss / len(valid_dataloader)
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        best_model_path = os.path.join(args.output_dir, model_name, f"prefix_tuning_seed_{args.seed}", "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        model.save_pretrained(best_model_path)

    logger.info(f"Epoch {epoch + 1} - Validation loss: {avg_validation_loss}")
    print(f"Epoch {epoch + 1} - Validation loss: {avg_validation_loss}")

    save_path = os.path.join(args.output_dir, model_name, f"prefix_tuning_seed_{args.seed}", f"epoch_{epoch + 1}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

# Save metrics
with open(f"{args.output_dir}/{model_name}/peak_memory.txt", "a") as f:
    f.write(f"prefix tuning: {str(peak_memory)}\n")

end_time = time.time()
training_time = end_time - start_time
with open(f"{args.output_dir}/{model_name}/training_time.txt", "a") as f:
    f.write(f"epoch: {args.num_epochs} prefix tuning: {str(training_time)}\n")