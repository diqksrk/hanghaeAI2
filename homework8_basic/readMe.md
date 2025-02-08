## 종합 Summary

### 학습 환경
1. 20000개의 데이터 중 100개를 Train, Validation DataSet으로 활용
2. Epoch는 30 Epoch까지만 반복
3. Tesla T4 GPU 2EA, GPUMemory 32GB, vCPU 32EA, Memory 160GB, [SSD]Disk 50GB 환경에서 실행
4. batch_size는 8, block size는 1024의 환경에서 실행

### 결과

| LoRA Lank | Final Loss | Trainning | Max GPU Memory (GB) |
|----------|----------|----------|----------|
| 8 | 9.895 | 250.672 | 2.6 |
| 128 | 10.099 | 283.758 | 2.7 |
| 256 | 10.157 | 303.91 | 2.9 |

### LoRa Lank 8
1. train / loss : https://wandb.ai/fivefingerfiv-hanghae999/Hanghae99/reports/train-loss-25-02-09-01-44-49---VmlldzoxMTI2NjYxMg
2. eval / loss : https://wandb.ai/fivefingerfiv-hanghae999/Hanghae99/reports/eval-loss-25-02-09-01-44-57---VmlldzoxMTI2NjYxNA

### LoRa Lank 128
1. train / loss : https://wandb.ai/fivefingerfiv-hanghae999/Hanghae99/reports/train-loss-25-02-09-01-44-14---VmlldzoxMTI2NjYwNg
2. eval / loss : https://wandb.ai/fivefingerfiv-hanghae999/Hanghae99/reports/eval-loss-25-02-09-01-44-30---VmlldzoxMTI2NjYwOA

### LoRa Lank 256
1. train / loss : https://wandb.ai/fivefingerfiv-hanghae999/Hanghae99/reports/train-loss-25-02-09-01-43-28---VmlldzoxMTI2NjU5OQ
2. eval / loss : https://wandb.ai/fivefingerfiv-hanghae999/Hanghae99/reports/eval-loss-25-02-09-01-43-51---VmlldzoxMTI2NjYwMw

### 깨달은점
1. 학습할때, 학습률이나 혹은 Drop Out의 비율도 마냥 낮다고 좋은것만은 아니였다. 마찬가지로 이 Rank가 늘어나면 더 많은 파라미터를 가져서 표현력이 상승하지만 Overfitting의 위험도가 증가한다.
2. 확실한건 더 많은 파라미터로 더 많은 계산을 수행하므로 메모리나 시간적에서는 점점 증가한다는 것이다.
3. 결국, 손실값이 최저로 도는 구간을 찾기 위해 여러가지 테스트가 필요하고 위에서 보았듯이 작은 값에서도 충분히 좋은 성능을 보이므로, 작은값부터 최적값을 찾기 위해 점점 올라가며 실험 하는것이 필요하다.

