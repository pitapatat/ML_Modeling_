
# [ML] 병원 폐업여부 예측 
**:rocket: library: pandas, seaborn, sklearn, imblearn(imbalanced-learn)**



### 1. 배경 및 목적

+ 병원의 재무제표 및 소재지, 병상수, 직원수 등의 데이터 분석을 통해 "계속 경영여부" 예측  
+ **(업데이트 - 22.06.05)** : 기존 데이터 분석([v1](https://github.com/pitapatat/ML_Modeling_/tree/main/business_closing))의 한계점 보완
```
+ 결측치 대체 방법 변경
+ feature engineering 전/후 결과 비교
+ 통계검정(t-test, chi-square test), ML model에 따른 특성 중요도 비교 및 가설 검증     
+ 하이퍼파라미터 튜닝을 통한 최적 모델 선택
+ 모델별 특징 및 결과 비교
```
----
### 2. 데이터셋 소개

+ 의료기관의 폐업여부가 포함된 2개년 재무정보와 병원정보 포함 데이터[DACON 연습용 데이터)]
+ 특징 : 타겟 데이터 불균형(약 9:1), 데이터의 수는 적지만 이상치가 많이 포함됨

----
### 3. 평가지표
+ DACON에서는 정확도(accuracy)를 기준으로 평가하였으나 타겟 데이터가 불균형(9:1) 하므로 적절하지 않다고 판단함
    + validation/test data에 대해 모든 라벨을 최빈값(=1)로 예측한 결과 >> **validation(정확도) = 0.9450, test(정확도) = 0.8730**
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
             <img src = "https://user-images.githubusercontent.com/83687942/172106592-90739c9e-6ece-4e18-a217-4df456be95e8.png" width="450" height="400">

      + 모델 학습 결과를 바탕으로 4개 모델(SVC, 랜덤포레스트, XGB, LGBM) 선택
      + gridresearchCV를 통한 하이퍼파라미터 튜닝 및 모델 최적화
      + voting classifier를 통한 최종 결과물 도출
   
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

#### 5-2. 모델 간 성능 비교
+ validation data 검증 결과의 f1-score/정확도(accuracy)/AUC score 비교 
 
   |모델|f1-score|accuracy|AUC score(base)|AUC score(최적화)|
   |:--:|:--:|:--:|:--:|:--:|
   |최빈값 예측|0.9718|0.9451|0.5| -|
   |SVM|0.9405 |0.8901| 0.6698|0.6768|
   |RandomForest|0.9660 | 0.9341|  0.8093|0.8605|
   |LGBM|0.9714 | 0.9451| 0.6814|0.7093|
   |XGB|0.9714 | 0.9451|0.6976|0.7547|
   

#### 5-3. 모델 간 특성중요도 비교 
+ 조회수와 높은 

----
### 6. 한계점 및 보완점
   
+ **<목적의 불명확성>**
   + test data의 타겟(label)데이터가 없음 
   + 예를 들어 광고 타겟층 예측, 광고 수익 예측등을 목표로 호응도 지표를 개발한다면 고려해야 할 변수는 (영상 조회수, 영상이 끊기는 시점, 코멘트 내용, 사용자 정보 데이터 등)으로 좀 더 구체적이고 명확해 질 수 있음. **<u>어떤 문제를 해결하기 위한 것인지를 명확히 할 필요가 있음</u>**

+ **<가설 설정 및 검증>**
   + 가설 검증을 위해서는 인기동영상/비인기동영상의 그룹군으로 구별하여 분석해야 하므로 인기동영상이 아닌 동영상의 데이터가 추가적으로 필요함
   
   + 애초에 가설 설정이 잘못되었으며 본 데이터를 통해서는 인기동영상의 특징과 지표(변수)간 관계에 대한 분석이 가능함. 이는 향후 동영상 제작 시 고려해볼 수 있는 요소로 활용될 수 있음. **<u>주어진 데이터로 해결할 수 있는 문제가 무엇인지 우선적으로 고려해야 함</u>**

+ **<호응도 지표 개발>**
   + 지표개발을 위해 유효 변수에 가중치를 부여하는 과정에서 통한 가중치 변환이 너무 단순하게 이뤄졌음
   
+ **<기타>** 
   + 인기동영상에 영향을 미치는 요인으로 데이터에서 주어진 변수 외 다른 요인(시기적 이슈-선거, 올림픽, 김장, 음박 출시 등)을 고려하지 못함

   + 채널(계정) 생성 시기 데이터를 추가하고, 인기동영상의 tag, description의 긍/부정 분석을 통해 어떤 특징이 있는지 살펴보면 좋을 것 같음
   

----
### 7. 업데이트('22.05)
   
+ _plotly_ 와 _streamlit_ 라이브러리를 활용하여 인터렉티브하게 시각화하고 대시보드 형태로 구현 



   







