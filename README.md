Multi-genre natural language inference(MNLI)를 해결했습니다. <br/>
[homework_advance_mnl](https://github.com/diqksrk/hanghaeAI2/blob/main/homework_advance_mnli.ipynb) 파일 참조

## Q1) 어떤 task를 선택하셨나요?
> MNLI 를 선택하였습니다.


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 모델 설계:
두 문장을 하나의 입력으로 결합하여 pre-trained DistilBERT 또는 non-pretrained DistilBERT를 fine-tuning하였습니다.
collate_fn 함수는 premise와 hypothesis를 결합하여 [SEP] 토큰으로 구분된 형태로 입력 데이터를 준비합니다. 분류에 최적화된 모델인 DistilBertForSequenceClassification 를 사용하여 문제를 해결했습니다.

**입력 형태 (shape):**  
- `input_ids`: (batch_size, max_seq_length)  
- `attention_mask`: (batch_size, max_seq_length)  
- `labels`: (batch_size)

**출력 형태 (shape):**  
- `logits`: (batch_size, num_labels)  # 유니크한 라벨이 3개였기에 3을 입력했습니다. (entailment, neutral, contradiction)  
- `loss`: ()


## Q3) 어떤 pre-trained 모델을 활용하셨나요?
> **활용한 모델:**  
- **Pre-trained:** `distilbert-base-uncased`  
- **Non-pretrained:** `DistilBERT`를 새롭게 초기화하여 사용하였습니다.  


## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> **결과 요약:**  
1. **Loss Curve**  
   - Pre-trained 모델: 초기 loss가 낮으며 빠르게 감소하였고, 수렴 속도가 빨랐습니다.
   - Non-pretrained 모델: 초기 loss가 높으며 감소 속도가 느렸습니다. 

2. **Accuracy**  
   - Pre-trained 모델: validation accuracy 약 80% 이상 도달하였습니다. 
   - Non-pretrained 모델: 초기 accuracy 낮고 validation 성능 약 60%에 머물렀습니다.

3. **Generalization 성능**  
   - Pre-trained 모델은 validation과 test dataset에서 더 높은 generalization 성능을 보였습니다. 
  
<img width="668" alt="image" src="https://github.com/user-attachments/assets/17066174-7dcd-4a4a-b890-98cb5b68f5a5" />

### 위의 사항들을 구현하고 나온 결과들을 정리한 보고서를 README.md 형태로 업로드
### 코드 및 실행 결과는 jupyter notebook 형태로 같이 public github repository에 업로드하여 공유해주시면 됩니다(반드시 출력 결과가 남아있어야 합니다!!) 
