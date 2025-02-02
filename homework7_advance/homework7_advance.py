import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

# wandb 초기화
wandb.init(project='Hanghae99')
wandb.run.name = 'instrucion_tunning_batch8'

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    dataset_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the dataset file, e.g., corpus.json'}
    )
    block_size: int = field(default=1024)
    num_workers: Optional[int] = field(default=None)

parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()  # log level을 INFO로 변경 

log_level = training_args.get_process_log_level()

# 각 로거들의 log level 설정
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

# corpus.json 파일을 데이터셋으로 불러옵니다.
raw_datasets = load_dataset("json", data_files=args.dataset_path)
# corpus.json은 보통 train split만 생성하므로, validation split이 없다면 80:20으로 분할합니다.
if "validation" not in raw_datasets:
    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.2)
    # 결과는 {'train': ..., 'test': ...} 형태로 생성되며, test split을 validation으로 사용합니다.
    raw_datasets["validation"] = raw_datasets.pop("test")

# 모델 및 토크나이저 불러오기
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# padding 토큰을 eos로 설정
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
)

# 만약 토크나이저의 vocabulary size가 모델의 embedding 크기보다 크다면 모델 resize
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# --- 토크나이즈 및 전처리 함수 정의 ---

def tokenize_function(examples):
    """
    각 예제에서 instruction, input, output 필드를 하나의 텍스트로 합칩니다.
    필요에 따라 prompt 형식을 수정할 수 있습니다.
    """
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        # 예시 포맷: "Instruction: ...\nInput: ...\nOutput: ..."
        text = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
        texts.append(text)
    return tokenizer(texts, truncation=True)

# 멀티프로세싱을 위한 main process 동기화
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=list(raw_datasets["train"].features.keys())
    )

# block_size 결정: 모델의 최대 길이를 고려하여 block_size를 설정합니다.
max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

def group_texts(examples):
    """
    토큰화된 텍스트들을 하나로 이어 붙인 뒤 block_size 단위로 자릅니다.
    이후 next token prediction을 위해 labels를 input_ids의 복사본으로 설정합니다.
    """
    # 모든 필드에 대해 리스트로 concat
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    
    # 전체 길이를 block_size로 나눌 수 있는 길이로 자릅니다.
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    
    # block_size 단위로 슬라이싱
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # next token prediction을 위해 labels 생성 (input_ids의 복사본)
    result["labels"] = result["input_ids"].copy()
    return result

def group_texts_for_eval(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 전체 길이가 block_size보다 작으면 원본 그대로 반환
    if total_length < block_size:
        return examples
    
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )
    
train_dataset = lm_datasets["train"]

validation_dataset = tokenized_datasets["validation"].map(
    group_texts_for_eval,
    batched=True,
    num_proc=args.num_workers,
)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,  # validation 데이터셋 추가
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# 체크포인트 로드 (있는 경우)
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint
    
# 학습 시작
train_result = trainer.train(resume_from_checkpoint=checkpoint)

# 모델 저장
trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
