
# [ML] 병원 폐업여부 예측 [:link:](https://github.com/pitapatat/ML_Modeling_/blob/main/business_closing/%5BML%5D_hospital_closing_prediction_v2.ipynb)
**:rocket: library: pandas, seaborn, sklearn, imblearn(imbalanced-learn)**



### 1. 배경 및 목적

+ 병원의 재무제표 및 소재지, 병상수, 직원수 등의 데이터 분석을 통해 "계속 경영여부" 예측  
+ **(업데이트 - 22.06.05)** : 기존 데이터 분석([v1](https://github.com/pitapatat/ML_Modeling_/blob/main/business_closing/v1/README_v1.md))의 한계점 보완
```
+ 결측치 대체 방법 변경
+ feature engineering 전/후 결과 비교
+ 통계검정(t-test, chi-square test), ML model에 따른 특성 중요도 비교 및 가설 검증     
+ 하이퍼파라미터 튜닝을 통한 최적 모델 선택
+ 모델별 특징 및 결과 비교
```
----
### 2. [데이터셋](https://dacon.io/competitions/official/9565/overview/description) 소개

+ 의료기관의 폐업여부가 포함된 2개년 재무정보와 병원정보 포함 DACON 연습용 데이터
+ 특징 : 타겟 데이터 불균형(약 9:1), 데이터의 수는 적지만 이상치가 많이 포함됨

----
### 3. 평가지표
+ DACON에서는 정확도(accuracy)를 기준으로 평가하였으나 타겟 데이터가 불균형(9:1) 하므로 적절하지 않다고 판단함
    + validation/test data에 대해 모든 라벨을 최빈값(=1)로 예측한 결과 >> **validation(정확도) = 0.9451, test(정확도) = 0.8730**
+ 이에 정확도(accuracy) 외 재현율(recall), f1 score, AUC score 등을 활용하여 종합적으로 평가하고자 함 

----
### 4. 분석 과정
   + 데이터 전처리 : 결측치 대체
      + **데이터의 값이 np.NaN인 경우 뿐만 아니라 0.0인 경우도 결측치로 간주함**
      + <u>병원 종류(instkind)는 병상수 기준으로(30개 미만/이상, 100개 미만/이상) 분류되었음을 전제로 함</u>
      + 병상수, 직원수, 수익 데이터를 바탕으로 병원 종류(instkind)의 결측치(3개) 보완 
      + 병원 종류(instkind)에 따른 그룹핑 후, 중앙값(median)을 사용하여 병상수(bedCount), 직원수(employee) 결측치 보완   
            <img src = "https://user-images.githubusercontent.com/83687942/172049021-938dacab-520a-44f9-8746-8d268c254d3e.png" width="450" height="250" >
            <img src = "https://user-images.githubusercontent.com/83687942/172049043-49780ebc-75b9-47e4-8664-17f77c799480.png" width="450" height="250" >  
      + feature engineering : 가설 검증을 위한 feature 생성
          <center><img src = "https://user-images.githubusercontent.com/83687942/172049509-488361ef-4904-4e02-8e12-1ed06f5a88a7.png" width="350" height="500"></center>
      
      + t-test, chi-square test를 통해 target(개/폐업) 그룹 간 유의미한 차이가 있는 feature 분석
         <center><img src = "https://user-images.githubusercontent.com/83687942/172104536-a6870b66-f7b6-46d1-8fed-8b15e266a13b.png" width="550" height="450"></center>
      + categorical features 인코딩 및 numerical features 정규화

   + 머신러닝 모델 학습 및 평가 
      + 로지스틱회귀, SVM, 결정트리, 랜덤포레스트, SGD/KNN/GradientBoosting/XGBM/LGBM 분류기 파이프라인(+SMOTE) 구축
         + ROC curve, AUC score를 사용하여 각 모델 별 최적의 threshold 적용
             <img src = "https://user-images.githubusercontent.com/83687942/172111798-a7178bfd-5f6c-4be2-923d-8a4212c0baf0.png" width="450" height="400">

      + 모델 학습 결과를 바탕으로 5개 모델(SVC, RandomForest, XGB, LGBM, GB)에 대한 gridresearchCV를 통한 하이퍼파라미터 튜닝 및 모델 최적화
          <center><img src = "https://user-images.githubusercontent.com/83687942/172112123-d955b4a6-a261-450f-a2b0-f69f143461cc.png" width="500" height="400"></center>
        
      + ~~voting classifier를 통한 최종 결과물 도출~~
   
   + 학습 모델에 따른 특성 중요도 분석 및 결과 해석
 
----
   
### 5. 가설 설정
   >  - 매출총이익률(=매출총이익/매출액)이 높을수록 폐업률이 낮을 것이다.
   >  - 영업이익(=매출총이익-(판관비+급여))이 높을수록 폐업률이 낮을 것이다.
   >  - 당기순이익이 높을수록 폐업률이 낮을 것이다.
   >  - 영업이익이 높은데 당기순이익이 낮다면 부채가 많아 이자비용이 많이 발생하는 것이므로 영업이익과 당기순이익이 반대 방향이면 폐업률이 높을 것이다.
   >  - ROE(투자금액으로 얼마만큼의 이익을 발생시키는지)가 높을수록 폐업률이 낮을 것이다.
   >  - ROA(이익 창출을 위해 자산을 얼마나 활용하는지)가 높을수록 폐업률이 낮을 것이다.
   >  - "총자산(순자산+부채총계)/순자산" 이 높고 ROE가 낮다면 폐업률이 높을 것이다. 
   >  - ownerChange == True 일수록 폐업률이 높을 것이다.
   >  - 부채비율(=부채/총자산)이 높을수록 폐업률이 높을 것이다. 
   
----
### 6. 분석 결과
#### 6-1. t-test, chi-square test 결과

+ 병원의 폐업 여부에 따라 유의미한 차이를 보이는 특성(features)
   + _매출액('revenue1'), 판매비와 관리비('sga1'), 급여('salary1'), 재고자산('inventoryAsset1/2'), 기타 비유동자산('OnonCAsset1/2'), 부채총계('debt1'), 비유동부채('NCLiabilities1'),매출총이익('tot_profit1'), 영업이익('ope_profit1'), 부채비율('debt_ratio1'), 대표자의 변경('ownerChange')_

#### 6-2. 모델 간 성능 비교
+ validation data 검증 결과 : base 모델의 성능이 가장 높음
 
   |모델|f1-score|accuracy|AUC score(base)|AUC score(최적화)|
   |:--:|:--:|:--:|:--:|:--:|
   |**최빈값 예측(base)**|**0.9718**|**0.9451**|**0.5**| -|
   |Logit Regression|0.9529| 0.9121|0.7581|0.7953|
   |SVM|0.9405 |0.8901| 0.6698|0.6768|
   |Random Forest|0.9660 | 0.9341|  0.8093|0.8605|
   |Gradient Boosting|0.9655|0.9341|0.6698|0.6628|
   |LGBM|0.9714 | 0.9451| 0.6814|0.7093|
   |XGB|0.9714 | 0.9451|0.6976|0.7547|
   
   

+ test data 검증 결과 : 학습 모델 중에선 Gradient Boosting 모델의 성능이 높았으나 base모델과 큰 차이를 보이지 못함

   |모델|accuracy(public/private)|비고|
   |:--:|:--:|:--:|
   |**최빈값 예측(base)**|**0.8730 / 0.8125**|-|
   |Logit Regression|0.7143 / 0.6719| -|
   |SVM|0.8413 / 0.7813|-|
   |Random Forest|0.8413 / 0.8125|-|
   |Gradient Boosting|0.8730 / 0.8438|-|
   |LGBM |0.8571 / 0.8281|-|
   |XGB|0.8413 / 0.8125 |-|
   |voting classifier1| 0.8412 / 0.8281 |SVM + Random Forest + XGB + LGBM|
   |voting classifier2| 0.8413 / 0.8281  |Random Forest + LGBM + XGB|
   |voting classifier3| 0.8413 / 0.8438 | GBM + XGB + LGBM |


#### 6-3. 모델 간 특성중요도 비교 
        * 특성중요도: 트리기반 분류기에서 각 특성을 모든 트리에 대해 평균 불순도 감소로 계산한 값

   <img src = "https://user-images.githubusercontent.com/83687942/172283175-049514ff-4249-41b2-99ed-e5b68216277b.png" width="230" height="400" ><img src = "https://user-images.githubusercontent.com/83687942/172287762-2a90e7fe-6d52-42de-b7af-a152d1369797.png" width="230" height="400"><img src = "https://user-images.githubusercontent.com/83687942/172287806-fc735ada-c7e8-444a-972e-4a1b1c7865ae.png" width="230" height="400" ><img src = "https://user-images.githubusercontent.com/83687942/172287852-42e13328-8ed4-4991-83ec-42ed5e58393d.png" width="230" height="400" >

   |특성중요도| Random Forest | XGB | LGBM | Gradient Boosting |
   |:--:|:--:|:--:|:--:|:--:|
   |1| 병원 종류 |병상수| ROE1|병원 종류|
   |2| 병상수| 총자산|ROA2|병상수|
   |3| 매출총이익률|ROA1|매출총이익 변화율|총자산|
   |4| 총자산|순자산|부채비율|ROA1|
   |5| ROE1|직원수 변동|총자산|순자산|
   |6| 시군구|매출원가|대표자변경|매출총이익 변화율|
   |7| 당기순이익|매출총이익|병상수|직원수 변동|
   |8| 장기미수금| 매출총이익 변화율|병원 종류|미수금|
   |9|ROE2|매출총이익률|부채비율|매출총이익|
   |10| 순자산|미수금|기타비유동자산|ROA2|

+ 모델 간 특성중요도에서 차이를 보이나 공통적으로 병원 종류, 병상수, 매출총이익(매출총이익률, 변화율), 자산(순자산, 총자산), ROA, ROE, 미수금(단기/장기미수금)의 특성이 중요한 것으로 나타남 
+ 이는 병원의 폐업에 영향을 주는 주요 요인으로 __병원의 규모와 매출총이익률, 자산, (총/순)자산 대비 당기순이익__변수를 고려할 수 있음을 의미함 


#### 6-4. 특이점
+ 하이퍼파라미터 튜닝을 통한 최적 모델로 학습하면 더 나은 결과를 보일것이라 생각했는데 성능면에서 기존(v1)보다 떨어진 모습을 보였고 결과적으로 최빈값으로 예측한 것보다 못한 성능을 나타냄. 하지만 base 모델은 validation date에 대한 AUC score가 0.5인 것으로 보아 좋은 모델이라고는 할 수 없음
+ test data의 label이 공개되지 않아 성능 향상을 위한 다양한 시도 및 지표 확인을 하지 못한 부분이 아쉬움
+ (try) 특성중요도 결과를 바탕으로 향후 특성수를 줄여서 다시 학습하는 방향을 시도해보고자 함   


----
### 7. 기존 분석의 한계점 보완사항

- ~~전처리 과정에 많은 시간을 투자하였으나 효과는 크지 않았다고 생각됨~~ 
    - 결측치 대체 방법 변경 후 동일한 feature engineering 적용 결과, 타겟을 예측하는데 영향을 미치는 유의미한 특성이 확인됨   

- ~~해당 가설이 유효했는지(특성중요도 등)에 대한 검증과 하이퍼파라미터 튜닝, CV 방법 등을 적용한 결과 보완 필요~~
    - 하이퍼파라미터 튜닝, 교차검증(CV), 각 모델의 특성중요도 분석을 통해 내용을 보완함 
 
- ~~ROC curve 등을 활용하여 최적의 임계점을 찾고자 하였으나 곡선의 모양이 비정상적인 모습을 보이고 threshold값이 모델별로 극단의 값을 보이는 등의 문제가 나타남. 문제의 원인 파악과 함께 개선 방향에 대한 고민 필요~~ 
   

----
### 8. 업데이트('22.06)



   







