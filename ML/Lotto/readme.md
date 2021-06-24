# Goal: 로또번호 예측하기
## 1차 시도: 피팅 실패
```
X: 5연속 로또 번호
Y: 45개 multi class E.g. 1, 3 => [0,1,0,1]
CNN + Flatten + BCEWithLogitsLoss
```
## 2차 시도: 피팅 실패 -> Colab pro를 결제해야하나...
```
순서의 상관관계 모델링
X1: multi encoding // E.g. 1, 3 => [0,1,0,1]
X2: stactical information // Frequency
Y: 45개 multi class E.g. 1, 3 => [0,1,0,1]
2차 시도(예정): DNN + [uni/bi]LSTM + BCEWithLogitsLoss
```

