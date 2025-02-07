---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---



# 2. 출발지의 비밀: 두 승객의 미스터리

## Embarked 피쳐 개요

Embarked 피쳐는 승객이 탑승한 항구를 나타내는 카테고리형 변수입니다. 총 세 개의 항구가 있습니다:

- C = Cherbourg, 프랑스
- Q = Queenstown, 아일랜드
- S = Southampton, 영국

## 타이타닉호의 항로

타이타닉호는 영국 Southampton에서 출발하여 프랑스 Cherbourg, 아일랜드 Queenstown을 거쳐 미국 뉴욕으로 향했습니다. 각 항구에서 승객들이 탑승했으며, 이는 당시 대서양 횡단의 일반적인 항로였습니다.

[여기에 항로를 보여주는 시각화 추가 예정]

## Embarked 결측치 분석

훈련 데이터셋에는 2명의 승객에 대한 Embarked 정보가 누락되어 있습니다. 이 승객들의 특성을 살펴보고, 어떤 항구에서 탑승했을지 추정해보겠습니다.

[결측치 승객 정보 분석 추가 예정]

## 결측치 처리

앞선 분석을 바탕으로 결측치를 적절한 값으로 대체하겠습니다.

## 인코딩

Embarked는 카테고리형 변수이므로, 머신러닝 모델이 이해할 수 있는 형태로 변환해야 합니다. 여기서는 원-핫 인코딩(One-Hot Encoding)을 사용하겠습니다.

## 모델 예측 결과

인코딩된 Embarked 피쳐를 사용하여 모델의 예측 성능이 어떻게 변화하는지 확인해보겠습니다.





📝 **프롬프트**

```

```

💻 **코드 & 실행결과**

```{code-cell}

```



본문....



Figure 추가

```{figure} ./images/11-1.png
---
width: 600px
---
결과파일 제출
```



본문....

