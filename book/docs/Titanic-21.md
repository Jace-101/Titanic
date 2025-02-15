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

💻 **준비 코드**

```{code-cell}
:tags: ["hide-input"]

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('gender_submission.csv')
```

<br>

# 2.1 성별(Sex)은 생존에 얼마나 큰 영향을 미쳤을까?

```{figure} ../images/21-1.png
---
width: 600px
---
```
*로즈는 떠다니는 나무 판자 위에서 생존할 수 있었지만, 잭은 차가운 바다 속에서 생을 마감했습니다. 이 장면은 당시 여성 승객들이 남성 승객들보다 더 높은 생존 기회를 가졌다는 역사적 사실을 극적으로 보여줍니다. (사진 출처: 영화 '타이타닉')*



<br>

앞서 우리는 수치형 변수만을 사용하여 첫 번째 예측 모델을 만들었습니다. 객실 등급, 나이, 요금 등의 정보로 63.397%의 정확도를 달성했죠. 하지만 영화 속 잭과 로즈의 운명이 암시하듯, 우리는 매우 중요한 정보를 하나 빠뜨렸습니다. 바로 성별입니다. 과연 성별 정보를 추가하면 예측 정확도가 얼마나 향상될까요?



## 성별과 생존율 사이의 관계

먼저 성별에 따른 생존율을 계산해보겠습니다. 앞서 배운 groupby와 agg 함수를 사용하면 쉽게 분석할 수 있습니다.

📝 **프롬프트**
```
성별에 따른 승객 수와 생존율을 계산해줘
```

💻 **코드 & 실행결과**
```{code-cell}
results = train.groupby('Sex')['Survived'].agg(['count', 'mean'])
results
```

실행 결과는 충격적입니다. 여성 승객의 생존율은 74.2%로 매우 높은 반면, 남성 승객의 생존율은 18.9%에 불과합니다. 이러한 극명한 차이는 타이타닉 침몰 당시 "여성과 어린이를 먼저 구하라(Women and children first)"는 원칙이 실제로 지켜졌음을 보여줍니다.



## 성별(Sex) 변수 인코딩하기

이제 성별 정보를 우리의 예측 모델에 추가해보겠습니다. 하지만 여기서 한 가지 문제가 있습니다. 지금까지 우리가 사용한 변수들은 모두 숫자였지만, Sex 변수는 'male'과 'female'이라는 문자열로 되어 있습니다. 대부분의 머신러닝 알고리즘은 숫자 데이터만 처리할 수 있기 때문에, 이 문자열을 숫자로 변환해야 합니다.

이러한 변환 과정을 '인코딩(encoding)'이라고 부릅니다. 가장 간단한 방법은 'male'을 0으로, 'female'을 1로 변환하는 것입니다. 이를 '레이블 인코딩(Label Encoding)'이라고 합니다.

📝 **프롬프트**
```
Sex 컬럼을 숫자로 변환하여 Gender라는 새로운 컬럼을 생성해줘
```

💻 **코드 & 실행결과**
```{code-cell}
# train 데이터에 Gender 피처 생성
train['Gender'] = train['Sex'].map({'male': 0, 'female': 1})

# test 데이터에도 Gender 피처 생성
test['Gender'] = test['Sex'].map({'male': 0, 'female': 1})
```

이 코드는 다음 작업을 수행합니다:

1. map 함수를 사용해 'male'을 0으로, 'female'을 1로 변환
2. 변환된 값을 새로운 'Gender' 컬럼에 저장
3. train과 test 데이터 모두에 동일한 변환 적용

<br>

변환이 제대로 되었는지 확인해보겠습니다:

📝 **프롬프트**
```
train에서 'PassengerId', 'Survived', 'Sex', 'Gender' 컬럼만 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
train[['PassengerId', 'Survived', 'Sex', 'Gender']].head()
```

결과를 보면 Sex 컬럼의 'male', 'female' 값들이 Gender 컬럼에서는 0과 1로 잘 변환된 것을 확인할 수 있습니다.



## 모델 학습과 예측

이제 Gender 변수를 포함하여 모델을 다시 학습시켜보겠습니다. 기존의 수치형 변수 리스트에 'Gender'를 추가합니다:

📝 **프롬프트**
```
기존의 수치형 변수 리스트에 'Gender'를 추가해줘
```

💻 **코드 & 실행결과**
```{code-cell}
inc_fts = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
inc_fts += ['Gender']  # 기존 리스트에 'Gender' 추가

print(inc_fts)
```

이제 1.3절에서 사용했던 모델 학습 코드를 실행하겠습니다:

```{code-cell}
# 데이터 준비
X = train[inc_fts]     # 선택한 특성들
y = train['Survived']  # 생존 여부
X_test = test[inc_fts] # 예측해야 할 데이터의 정보들

# 학습/검증 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 성능 평가
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation Score: {accuracy:.5f}")

# 테스트 데이터 예측 및 저장
y_test_pred = model.predict(X_test)
submission['Survived'] = y_test_pred
submission.to_csv('titanic_pred.csv', index=False)
```

:::{note}
**모델 학습 코드의 재사용**

위의 모델 학습 코드는 앞으로 여러 섹션에서 반복적으로 사용됩니다. 다음 섹션부터는 이 코드가 "준비 코드"에 포함되어 있으므로, 매번 입력할 필요 없이 바로 실행 결과만 확인하면 됩니다.

준비 코드에 포함되는 내용:
1. 데이터 준비 (X, y, X_test 생성)
2. 학습/검증 데이터 분할
3. Random Forest 모델 학습
4. 성능 평가 및 예측 결과 저장

이는 마치 요리 책에서 기본 반죽을 미리 준비해두고 다양한 빵을 만드는 것과 같습니다.
:::

Random Forest 모델을 사용하여 학습하고, 검증 데이터로 성능을 평가한 결과, Validation Score가 0.82123으로 나왔습니다. 이는 이전 모델(0.73184)보다 크게 향상된 결과입니다.

실제 test 데이터에 대한 예측 결과를 캐글에 제출했더니 0.73444의 점수를 얻었습니다. 418명의 test 데이터 중 307명의 생존 여부를 맞혔다는 의미입니다. 수치형 변수만 사용했을 때의 점수가 0.63397(265명)이었던 것과 비교하면, Gender 변수를 추가함으로써 42명을 더 정확하게 예측할 수 있게 되었네요.

지금까지의 결과를 표로 정리하면 다음과 같습니다:

| 버전 | 피쳐 개수 | Val. Score | Public Score | 맞은 사람 수 |          설명           |
| :--: | :-------: | :--------: | :----------: | :----------: | :---------------------: |
| 1.3  |     5     |  0.73184   |   0.63397    |    265명     | 5개의 numeric 피쳐 사용 |
| 2.1  |     6     |  0.82123   |   0.73444    |    307명     |    Gender 피쳐 추가     |

단 하나의 변수를 추가했을 뿐인데 정확도가 10%p 이상 향상되었습니다. 이는 성별이 생존 여부를 예측하는데 매우 중요한 요소였음을 다시 한 번 확인시켜주는 결과입니다.

다음 섹션에서는 승객들의 탑승 항구(Embarked) 정보를 추가하여 모델을 더욱 개선해보도록 하겠습니다. 과연 이 정보는 예측 정확도를 얼마나 향상시킬 수 있을까요?



:::{admonition} 데이터 속 숨은 이야기 
:class: seealso

**"Women and children first"의 유래: 버켄헤드 규율**

타이타닉호에서 지켜진 "여성과 어린이 먼저"라는 원칙은 사실 그보다 60년 전인 1845년의 한 사건에서 비롯되었습니다. 영국 군함 HMS 버켄헤드(Birkenhead)가 남아프리카 해안에서 좌초했을 때의 일입니다.

침몰이 시작되자 함장은 제한된 구명보트에 여성과 어린이들을 먼저 태웠습니다. 그리고 남성 군인들은... 놀랍게도 침몰하는 배 위에 정렬해 서있었다고 합니다. 그들은 구명보트로 달려가는 혼란스러운 상황이 여성과 어린이의 탈출을 방해할 수 있다고 판단했기 때문입니다.

이 이야기는 빅토리아 시대 영국의 기사도 정신을 상징하는 사건이 되었고, "버켄헤드 규율(Birkenhead Drill)"이라는 이름으로 이후 해상 구조의 기본 원칙이 되었습니다.

타이타닉호의 경우 이 원칙이 잘 지켜져서 우리가 방금 확인한 것처럼 여성의 생존율(74.2%)이 남성(18.9%)보다 훨씬 높았습니다. 영화에서 보여준 여러 장면들, 특히 구명보트에 탑승하는 순서를 다루는 장면들은 이러한 역사적 맥락을 정확히 반영한 것이었죠.
:::

<br>
