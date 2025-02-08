import os
import sys
import torch
import wandb
import logging
import datasets
import transformers
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType

from typing import Optional
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator
)
import copy
import shutil

# wandb 초기화
wandb.init(project='Hanghae99')
wandb.run.name = 'homework8_basic_rora256'

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    block_size: int = field(default=1024)
    no_fp16: bool = field(default=False, metadata={"help": "Disable FP16 training."})
    num_workers: Optional[int] = field(default=None)

parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# 이전 체크포인트 로드 방지를 위해 resume 옵션 제거
training_args.resume_from_checkpoint = None
# output_dir가 존재하면 삭제해서 기존 체크포인트 파일들을 제거합니다.
if os.path.exists(training_args.output_dir):
    shutil.rmtree(training_args.output_dir)

# 로깅 설정
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

# 데이터셋 로드
raw_datasets = load_dataset(args.dataset_name)
# validation 스플릿이 없으면 train 스플릿에서 80:20 비율로 분할하여 생성합니다.
if "validation" not in raw_datasets:
    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.2)
    raw_datasets["validation"] = raw_datasets.pop("test")

# 모델 및 토크나이저 로드
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# global tokenizer는 미리 PAD 토큰이 추가되어 있어야 함 (워커에게 전달하기 전)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# 기본 모델을 base_model으로 저장하여 각 LoRA 실험마다 deepcopy하여 사용합니다.
base_model = model

# 토큰화 함수 (batched 입력 처리)
def tokenize_function(examples, tokenizer):
    texts = []
    for inst, inp in zip(examples["instruction"], examples["input"]):
        text = inst + ("\n" + inp if inp else "")
        texts.append(text)
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 데이터셋 전처리 (워커에 수정된 토크나이저를 직접 전달)
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # num_proc=1로 설정하여 멀티프로세싱 문제를 피할 수 있습니다.
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=raw_datasets["train"].column_names
    )

train_dataset = tokenized_datasets["train"].select(range(100))
validation_dataset = tokenized_datasets["validation"].select(range(100))

# LoRA rank 실험 설정
lora_r_values = [8, 128, 256]
results_table = wandb.Table(columns=["LoRA Rank", "Final Loss", "Training Time (s)", "Max GPU Memory (GB)"])

for lora_r in lora_r_values:
    logger.info(f"Starting training with LoRA rank: {lora_r}")
    
    # 모델을 새로 로드합니다.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=args.torch_dtype
    )
    
    # global tokenizer는 계속 재사용 (PAD 토큰이 추가되어 있으므로 모델의 임베딩 크기를 재조정)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
    # 모델 이름에 따라 target_modules 설정 (예: openai-gpt의 경우)
    if "gpt" in args.model_name_or_path.lower():
        target_modules = ["c_attn", "c_proj"]
    elif "opt" in args.model_name_or_path.lower():
        target_modules = ["q_proj", "v_proj"]
    else:
        # 그 외 모델의 경우 적절한 target_modules 설정
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_token = name.split('.')[-1]
                if "dropout" in last_token.lower():
                    continue
                target_modules.append(last_token)
        target_modules = list(set(target_modules))
        if "lm_head" in target_modules:
            target_modules.remove("lm_head")
        if not target_modules:
            raise ValueError("No target modules found in the base model. Please specify target modules manually.")
    
    logger.info(f"Using target modules: {target_modules}")

    # LoRA 설정 및 적용
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)

    # Trainer 설정 및 학습 진행
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=training_args,
        data_collator=default_data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=None)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    # trainer.save_state()

    max_memory = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1) if torch.cuda.is_available() else 0.0
    print(f"LoRA Rank {lora_r} - Max GPU Memory Used: {max_memory} GB")
    print('Max Alloc:', round(torch.cuda.max_memory_allocated(0) / 1024**3, 1), 'GB')

    wandb.log({
        "LoRA Rank": lora_r,
        "Final Loss": metrics["train_loss"],
        "Training Time (s)": metrics["train_runtime"],
        "Max GPU Memory (GB)": max_memory
    })
    results_table.add_data(lora_r, metrics["train_loss"], metrics["train_runtime"], max_memory)

wandb.log({"LoRA Experiments": results_table})
wandb.finish()
