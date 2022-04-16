# [ML] 병원 폐업여부 예측

:rocket: **library: pandas, seaborn, sklearn, imblearn(imbalaced-learn), shap**

### 1. 프로젝트 개요
```
- 배경 : 기존의 금융기관은 병원 대출 시 신용점수 또는 담보물 위주로 평가 
- 목적 : 병원의 재무제표 분석을 통해 "상환 기간동안의 계속 경영여부"를 예측하여 대출 기준 미충족자에게 금융 기회를 제공
- 과정 : 가설 설정 - 데이터 전처리/시각화 - feature engineering - 모델 학습 - (데이터 불균형 해결) - 결과 해석
- 평가지표 : f1 score
```

### 2. [데이터셋](https://dacon.io/competitions/official/9565/overview/description) 
```
- DACON 연습용 데이터(train: 301개, test: 127개)
- 의료기관의 폐업여부가 포함된 2개년 재무정보와 병원 기본정보(지역/종류/병상수 외) 포함 데이터 
- 데이터 전처리/분리 : 결측치 확인 , 중복 제거 후 train: validation = 8:2 로 나누어 학습 
```
```
- 특징 : ① 데이터 개수가 적음 ② 타겟 데이터 불균형(

### 3. 모델 학습
* 목표: pre-trained model을 사용하여 모델 훈련 후, 하이퍼파라미터 튜닝을 통해 성능 향상
* **ELECTRA model**
   <center><img width = '700' height = '150' src ='https://user-images.githubusercontent.com/83687942/163327105-67d133c7-395b-4919-9d76-36fa17b1b9b7.jpg'></center>
  
  * Masked Language Model을 개선시킨 RTD(Replaced Token Detection)방식으로 학습하는 모델
  * RTD : masking된 단어를 generator를 통해 다른 단어로 대체한 후 discriminator로 원본과 대조하여 학습하는 방식
  
* 학습모델 : [KoELECTRA](https://github.com/monologg/KoELECTRA/tree/master/finetune)
   
    |model|batch_size | train_steps| learning_rate | max_seq_len|
    |-|-|-|-|-|
    |base-v3|256|1.5M|2e-4|512|



### 4. 하이퍼파라미터 튜닝

|hyperparameters|&nbsp;base&nbsp; |&nbsp;best&nbsp;| reason|
|:-:|:-:|:-:|-|
|learning_rate|5e-5|5e-5|learning rate을 증가시키면 train loss가 줄어들지 않음
|epochs|10|20| epoch 2배 조정시 성능이 크게 향상되었으나 그 이상은 투입 자원 대비 성능향상이 미미함
|warmup_proportion|0.0|0.2|0.1씩 증가시킨 결과 0.2에서 가장 높은 성능향상
|weight_decay|0.0|0.0|epoch 증가에 따라 weight decay를 함께 늘리면 성능이 향상되었으나 best는 아님
|batch_size|32|32|batch size 증가시 out of memory 에러가 발생함
|max_seq_len|120|120|학습데이터의 문장 평균 값을 고려하여 80으로 조정하였으나 성능이 하락함




### 5. 학습 결과

<center><img width = '400' height = '300' src = 'https://user-images.githubusercontent.com/83687942/163331560-62bbf38a-e58a-4a1f-8339-14938750726a.jpg'></center>
<center><img width = '400'  height = '300' src ='https://user-images.githubusercontent.com/83687942/163331659-0f273aa6-fd48-4986-8733-8bbc785b1cde.jpg'></center>

* validation loss 기준 (loss 증가 직전) epochs=2는 훈련량 부족으로 높은 성능을 보이지 않음
* eopchs=16과 19의 f1 score 비교 결과 check point는 "epochs:19" 설정

|- | base model | best hyperparameter tuned model|
|:-:|:-:|:-:|
|&nbsp;&nbsp;&nbsp;&nbsp;f1 score&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;0.8641035996&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|**0.87802585193**|
|pearson correlation|&nbsp;&nbsp; | 0.93091781336|




### 6. [FAST API](https://github.com/seyeonjungGit/team1_API) 구현

<center><img width = '600' height = '400' src = 'https://user-images.githubusercontent.com/83687942/163333897-11740a11-5a0d-49b5-8504-d95452b7a3d0.jpg'></center>



### 7. 과제 담당 역할
```
- 모델 조사 및 논문 리서치(SBERT)
- 하이퍼파라미터 튜닝 
```



### 8. 한계점 및 보완점
```
- 하이퍼파라미터 튜닝의 자동화 필요
- 추후 토큰화 방식의 다양화(형태소 임베딩 등)를 통한 모델의 성능 비교 
- (개인) 오픈소스를 활용한 논문의 코드 구현 및 모듈화를 위한 역량 개발 필요
```
