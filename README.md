# level1_bookratingprediction-recsys-01
level1_bookratingprediction-recsys-1 created by GitHub Classroom

## Book Rating Prediction

### 프로젝트 개요

책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점 데이터를 활용하여 사용자가 새로운 책에 대해 부여할 평점을 예측함으로써 소비자의 책 선택에 도움이 될 수 있도록 하는 것이 프로젝트의 목표입니다.

### 협업 툴 및 기술 스택

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dd46a5a1-0ccd-4769-9d57-a5c317539855/Untitled.png)

### 프로젝트 결과

- public 5위
    - RMSE: 2.1126
    - Vanilla CNN_FM 대비 약 9.7% 성능 향상
- private 3위
    - RMSE: 2.1077
    - Vanilla CNN_FM 대비 약 9.8% 성능 향상

### 솔루션 모델

총 11개의 모델들에 대해서 stacking 앙상블을 수행하였습니다.

- 베이스라인 모델
    - FFM, DCN, NCF, DeepCoNN
- Hybird DL 모델
    - 3 layer CNN context FM:  베이스라인에서 제공된 CNN_FM에서 레이어를 추가하고, 컨텍스트 데이터를 함께 고려할 수 있도록 하여 더 좋은 성능에 더 적은 에폭수로 도달하도록 함
    - FFDCN: FFM과 DCN을 병합한 하이브리드 모델, FFM의 출력과 DCN의 출력을 concat한 후, linear layer의 출력으로 평점을 예측
    - DeepCoNN_CNN: DeepCoNN모델의 FM layer에 이미지 vector를 추가한 모델
- Tree-based ML 모델
    - Catboost (Top 1, 2 Rated): 트리 기반 모델로 범주형 dataset에서 예측 성능이 우수한 모델, 단독 성능도 좋으며 앙상블시 높은 성능 향상을 보임.
    - LightGBM: 트리 기반 모델로 Leaf-wise 방식을 사용해 학습속도가 빠르다는 장점을 갖는 모델. 단독 성능은 좋지 않았던 반면 앙상블시 좋은 성능을 보임.

### 프로젝트 구성원 및 역할

| (사진) | (사진) | (사진) | (사진) | (사진) |
| --- | --- | --- | --- | --- |
| https://github.com/kCMI113 | https://github.com/DyeonPark | https://github.com/alstjrdlzz | https://github.com/2jun0 | https://github.com/juhyein |
- **강찬미**: 필드 결측치 처리, Tree-based ML 모델 구현, 하이퍼파라미터 튜닝
- **박동연**: 필드 결측치 처리, CNN_FM 모델 구조 변경, 하이퍼파라미터 튜닝
- **서민석**: 실험 환경 구축, 교차검증, OOF, 배치 샘플러, Content-based filtering 구현, 하이퍼파라미터 튜닝
- **이준영**: 실험 환경 구축, stacking 구현, Hybird DL 모델 구현, 하이퍼파라미터 튜닝
- **주혜인**: 필드 결측치 처리, Optuna구현, early stopping구현, 하이퍼파라미터 튜닝

### **[랩업리포트 (pdf)](**./BoostCamp5%20Project1%20Wrapup%20Report.pdf**)**

프로젝트 진행에 관한 더 자세한 내용은 랩업리포트 pdf 파일로 확인할 수 있습니다
