from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer
import torch
import wandb

# corpus.json 파일로부터 데이터셋 로드
dataset = load_dataset("json", data_files="corpus.json")["train"]

# 데이터셋 분할 (70% 학습, 20% 검증, 나머지 테스트)
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

from peft import LoraConfig, get_peft_model

# LoRA 설정 (모델 아키텍처에 맞게 target_modules 등을 조정하세요)
lora_config = LoraConfig(
    r=256,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 예시: 모델에 따라 달라질 수 있습니다.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Llama‑3.2‑1B 모델 로드 (양자화 및 device_map 적용)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B"
).to("mps")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# Llama 계열 모델에 pad_token이 없을 경우, eos_token을 pad_token으로 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 전처리: instruction과 output을 합쳐 prompt 형식 적용
def format_prompt(example):
    prompt = f"### 질문: {example['instruction']}\n### 답변: {example['output']}"
    return {"text": prompt}

# 토크나이즈 함수
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True  # 길이 초과 시 자르기
    )

# 전처리 및 토크나이즈 (batched=True)
train_dataset = train_dataset.map(format_prompt).map(tokenize_function, batched=True)
val_dataset = val_dataset.map(format_prompt).map(tokenize_function, batched=True)
test_dataset = test_dataset.map(format_prompt).map(tokenize_function, batched=True)

# 학습 인자 설정
# 모델이 4-bit 양자화 및 fp16을 사용하므로 Trainer의 fp16 옵션을 그대로 사용합니다.
training_args = TrainingArguments(
    output_dir="./llama3.2-1B-sft",   # fine-tuning 결과 저장 디렉토리
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=False,
    save_total_limit=2,
    push_to_hub=False,
    greater_is_better=False,
    load_best_model_at_end=False,
    report_to="wandb",
    run_name="week7_advanced_llama_finetuning",
    metric_for_best_model="eval_loss",
)

# SFTTrainer 생성
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

results_table = wandb.Table(columns=["LoRA Rank", "Final Loss", "Training Time (s)", "Max GPU Memory (GB)"])
# 학습 시작
train_result = trainer.train(resume_from_checkpoint=None)

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
# trainer.save_state()

max_memory = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1) if torch.cuda.is_available() else 0.0
print(f"LoRA Rank 256 - Max GPU Memory Used: {max_memory} GB")
print('Max Alloc:', round(torch.cuda.max_memory_allocated(0) / 1024**3, 1), 'GB')

wandb.log({
    "LoRA Rank": 256,
    "Final Loss": metrics["train_loss"],
    "Training Time (s)": metrics["train_runtime"],
    "Max GPU Memory (GB)": max_memory
})
results_table.add_data(256, metrics["train_loss"], metrics["train_runtime"], max_memory)