import os
import json
import logging
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
import traceback



def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from the provided JSON file."""
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")


def setup_logger(log_dir: str, log_level: str) -> logging.Logger:
    """
    Setup logging configuration with both file and stream handlers.
    Logs are saved to log_dir/training.log and also printed to the console.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(log_level.upper())

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(stream_handler)

    return logger


def load_model_and_tokenizer(config: dict):
    """
    Load model and tokenizer based on the configuration.
    Uses FastLanguageModel from unsloth.
    """
    try:
        model_name = config["model"]["name"]
        max_seq_length = config["model"]["max_seq_length"]
        dtype = config["model"]["dtype"]
        load_in_4bit = config["model"]["load_in_4bit"]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")


def format_prompts(examples, EOS_TOKEN: str, prompt_template: str):
    """
    Format input examples to a Trump-style prompt.
    
    Assumes the examples have columns:
      - "instruction": a prompt or topic.
      - "output": the corresponding Trump speech text.
    
    The formatted text will append the tokenizer's EOS_TOKEN.
    """
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for ins, out in zip(instructions, outputs):
        formatted_text = prompt_template.format(ins, out) + EOS_TOKEN
        texts.append(formatted_text)
    return {"text": texts}


def setup_lora(config: dict, model):
    """Configure LoRA settings for the model if enabled."""
    try:
        if config["lora"].get("enabled", False):
            model = FastLanguageModel.get_peft_model(
                model,
                r=config["lora"]["rank"],
                target_modules=config["lora"]["target_modules"],
                lora_alpha=config["lora"]["alpha"],
                lora_dropout=config["lora"]["dropout"],
                bias=config["lora"]["bias"],
                use_gradient_checkpointing="unsloth",  # memory optimization
                random_state=config["training"].get("seed", 3407),
                use_rslora=False,
                loftq_config=None,
            )
        return model
    except Exception as e:
        raise ValueError(f"Error configuring LoRA: {e}\n{traceback.format_exc()}")


def create_trainer(model, tokenizer, train_dataset, test_dataset, config: dict):
    """Create and return an SFTTrainer object using the provided datasets and training arguments."""
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field="text",
            max_seq_length=config["model"]["max_seq_length"],
            dataset_num_proc=config["training"].get("dataset_num_proc", 2),
            packing=False,  # speeds up training for short sequences
            args=TrainingArguments(
                per_device_train_batch_size=config["training"]["train_batch_size"],
                gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
                warmup_steps=config["training"]["warmup_steps"],
                max_steps=config["training"]["max_steps"],
                learning_rate=config["training"]["learning_rate"],
                fp16=not is_bfloat16_supported() and config["model"].get("fp16", False),
                bf16=is_bfloat16_supported() or config["model"].get("bf16", False),
                logging_steps=config["training"].get("logging_steps", 500),
                optim="adamw_8bit",
                weight_decay=config["training"]["weight_decay"],
                lr_scheduler_type="linear",
                adam_epsilon=config["training"].get("adam_epsilon", 1e-8),
                max_grad_norm=config["training"].get("max_grad_norm", 1.0),
                seed=config["training"].get("seed", 3407),
                output_dir=config["output"]["output_dir"],
                report_to="none",  # Disable external logging (e.g. WandB)
                save_steps=config["output"].get("save_model_steps", 1000),
            ),
        )
        return trainer
    except Exception as e:
        raise ValueError(f"Error creating trainer: {e}\n{traceback.format_exc()}")


def save_model_and_tokenizer(model, tokenizer, save_directory: str):
    """Save the model and tokenizer to the specified directory."""
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)


def fine_tune_model(config: dict, logger: logging.Logger):
    """Fine-tune the model using the provided configuration."""
    # Set seed for reproducibility
    set_seed(config["training"].get("seed", 3407))

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Load preprocessed data (CSV file with Trump speeches)
    logger.info("Loading preprocessed data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_data = pd.read_csv(os.path.join(script_dir, "data/preprocessed_trump.csv"))
    EOS_TOKEN = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    # Format the data with the Trump-style prompt template
    logger.info("Formatting prompts...")
    dataset = Dataset.from_pandas(preprocessed_data)
    dataset = dataset.map(
        lambda examples: format_prompts(examples, EOS_TOKEN, config["prompt"]["train_template"]),
        batched=True,
        num_proc=config["training"].get("dataset_num_proc", 2)
    )

    # Split the dataset into train and test sets
    logger.info("Splitting dataset into train and test sets...")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=config["training"].get("seed", 3407))
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # Setup LoRA configuration if enabled
    logger.info("Configuring LoRA...")
    model = setup_lora(config, model)

    # Create Trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(model, tokenizer, train_dataset, test_dataset, config)

    # Train the model
    logger.info("Starting training...")
    trainer_stats = trainer.train()

    # Save the final model and tokenizer locally
    save_directory = config["output"]["output_dir"]
    logger.info(f"Saving model and tokenizer to {save_directory}...")
    save_model_and_tokenizer(model, tokenizer, save_directory)

    return trainer_stats


def fine_tune():
    """Main function to run the fine-tuning process."""
    # Determine config file path relative to this script’s location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config/config.json")
    #config_path = os.path.join(os.getcwd(), 'config/config.json')
    

    # Load configuration
    config = load_config(config_path)

    # Setup logger
    logger = setup_logger(config["logging"]["log_dir"], config["logging"]["log_level"])
    logger.info("Starting fine-tuning process...")

    # Fine-tune model
    trainer_stats = fine_tune_model(config, logger)
    logger.info(f"Training stats: {trainer_stats}")
    logger.info("Fine-tuning completed successfully.")


if __name__ == '__main__':
    fine_tune()
