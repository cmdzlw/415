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
    IA3Config,
    PeftType
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)

from tqdm import tqdm


# ====== 新增：文本清洗函数 ======
def clean_text(text):
    """安全清洗文本：处理 NaN、None、列表、非字符串、换行符等"""
    if isinstance(text, list):
        text = " ".join(str(x) for x in text if pd.notna(x))
    elif pd.isna(text) or text is None:
        text = ""
    else:
        text = str(text)
    return text.replace("\n", " ").replace("\r", " ").strip()
# ==============================


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="./LLMs/graphcodebert-base")
    parser.add_argument("--train_file", type=str, default="dataset/牛逼/train.csv")
    parser.add_argument("--valid_file", type=str, default="dataset/牛逼/valid.csv")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_train_samples", type=int, choices=[100, 200, 500, 1000], default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="Adamw")
    parser.add_argument("--should_log", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_name_or_path.split("/")[-1]

    # Setup logging
    log_dir = os.path.join(args.output_dir, model_name, f"ia3_seed_{args.seed}")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"train_max_samples_{args.max_train_samples}_log.txt")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.should_log else logging.WARNING,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file_path)
        ]
    )
    logger = logging.getLogger(__name__)

    content_write = "=" * 50 + "\n"
    content_write += "IA3\n"
    content_write += f"seed: {args.seed}\n"
    content_write += f"model_name_or_path: {args.model_name_or_path}\n"
    content_write += f"train_file: {args.train_file}\n"
    content_write += f"valid_file: {args.valid_file}\n"
    content_write += f"max_seq_length: {args.max_seq_length}\n"
    content_write += f"batch_size: {args.batch_size}\n"
    content_write += f"num_epochs: {args.num_epochs}\n"
    content_write += f"max_train_samples: {args.max_train_samples}\n"
    content_write += f"learning_rate: {args.learning_rate:.0e}\n"
    content_write += f"optimizer: {args.optimizer}\n"
    content_write += f"should_log: {args.should_log}\n"
    content_write += f"output_dir: {args.output_dir}\n"
    content_write += "=" * 50 + "\n"
    logger.info(content_write)
    print(content_write)

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    model_lower = args.model_name_or_path.lower()

    # Set PEFT config
    peft_config = IA3Config(
        task_type="SEQ_CLS",
        target_modules=LORA_IA3_TARGET_MODULES[model_name]["target_modules_ia3"],
        feedforward_modules=LORA_IA3_TARGET_MODULES[model_name]["ff_modules"]
    )

    # Load tokenizer
    padding_side = PADDING_SIDE.get(model_name, "right")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and clean datasets
    train_df = load_trainset(args.train_file, max_train_samples=args.max_train_samples, seed=args.seed)
    eval_df = pd.read_csv(args.valid_file)

    # ====== 清洗文本 ======
    train_df["text"] = train_df["text"].apply(clean_text)
    eval_df["text"] = eval_df["text"].apply(clean_text)
    # =====================

    trainset = Dataset.from_pandas(train_df)
    evalset = Dataset.from_pandas(eval_df)
    datasets = DatasetDict({'train': trainset, 'validation': evalset})

    num_labels = len(train_df["label"].unique())

    # Handle max_seq_length
    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_length)

    # Safe column removal
    available_columns = datasets["train"].column_names
    remove_cols = [col for col in ["text", "text_label"] if col in available_columns]

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=False
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors="pt", padding=True, max_length=max_seq_length)

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
    logger.info(f"IA3 trainable parameters: {model.get_nb_trainable_parameters()}")
    model.print_trainable_parameters()

    # Fix pad token for certain models
    if "deepseek" in model_lower or "starcoder" in model_lower:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    # Optimizer & scheduler
    if args.optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.06 * len(train_dataloader) * args.num_epochs),
        num_training_steps=len(train_dataloader) * args.num_epochs
    )

    if use_cuda:
        model.cuda()

    # Training loop
    best_validation_loss = float("inf")
    peak_memory = 0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        # Training
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
            if use_cuda:
                batch = {k: v.cuda() for k, v in batch.items()}
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

            if use_cuda:
                current_memory = torch.cuda.max_memory_allocated()
                peak_memory = max(peak_memory, current_memory)

        progress_bar_train.close()
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.6f}")
        print(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.6f}")

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
            if use_cuda:
                batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                total_validation_loss += outputs.loss.item()
            if step % 5 == 0:
                progress_bar_valid.update(5)

        progress_bar_valid.close()
        avg_validation_loss = total_validation_loss / len(valid_dataloader)
        logger.info(f"Epoch {epoch + 1} - Validation loss: {avg_validation_loss:.6f}")
        print(f"Epoch {epoch + 1} - Validation loss: {avg_validation_loss:.6f}")

        # Save best model
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            best_model_path = os.path.join(log_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            logger.info(f"Best model saved to {best_model_path}")

        # Save epoch checkpoint
        epoch_path = os.path.join(log_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_path, exist_ok=True)
        model.save_pretrained(epoch_path)

    # Save stats
    training_time = time.time() - start_time
    info_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(info_dir, exist_ok=True)

    with open(os.path.join(info_dir, "peak_memory.txt"), "a") as f:
        f.write(f"ia3: {peak_memory}\n")

    with open(os.path.join(info_dir, "training_time.txt"), "a") as f:
        f.write(f"epoch: {args.num_epochs} ia3: {training_time:.2f}\n")

    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Peak memory usage: {peak_memory} bytes")


if __name__ == "__main__":
    main()