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



# 1. 성별(Sex)을 추가하면 생존 예측의 정확도가 얼마나 올라갈까?

타이타닉 영화에서 남자 주인공인 잭은 죽었고, 여자 주인공인 로즈는 구조되었습니다. 실제 사고에서도 그랬을까요? train 데이터에서 성별에 따른 생존비율을 확인해봅시다.

```{figure} ./images/21-1.png
---
width: 600px
---
영화 '타이타닉'의 한 장면
```

*로즈는 나무 위에서 생존할 수 있었지만, 잭은 차가운 바다 속에서 생을 마감했습니다. 이 장면은 당시 여성 승객들이 남성 승객들보다 더 높은 생존 기회를 가졌다는 역사적 사실을 극적으로 보여줍니다.*



💻 **준비 코드**

```{code-cell}
:tags: ["hide-input"]

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('gender_submission.csv')
```



## 성별과 생존율 사이의 관계

먼저 성별에 따른 생존율을 계산하는 코드를 요청해보겠습니다.

📝 **프롬프트**

```
성별에 따른 승객 수와 생존율을 계산해줘
```

💻 **코드 & 실행결과**

```{code-cell}
results = train.groupby('Sex')['Survived'].agg(['count', 'mean'])
results
```


코드를 살펴보면, groupby() 함수를 사용하여 'Sex' 컬럼을 기준으로 데이터를 그룹화했습니다. 그리고 agg() 함수를 사용하여 각 그룹별로 두 가지 계산을 수행했습니다:

1. count: 각 성별의 승객 수를 세어줍니다.
2. mean: Survived 컬럼의 평균값을 계산합니다. Survived는 0(사망)과 1(생존)로 구성되어 있으므로, 평균값은 곧 생존율을 의미합니다.

결과를 보면 놀라운 사실을 발견할 수 있습니다. 여성 승객의 생존율은 약 74.2%로 매우 높은 반면, 남성 승객의 생존율은 약 18.9%에 불과합니다. 이는 타이타닉 침몰 당시 "여성과 어린이를 먼저 구하라(Women and children first)"는 원칙이 실제로 지켜졌음을 보여줍니다. 영화의 설정은 이러한 역사적 사실을 정확하게 반영한 것이었네요.



## 성별(Sex) 변수 인코딩하기

챕터 1에서는 수치형 변수들만 사용하여 생존자를 예측해보았습니다. 하지만 결과는 그리 좋지 않았죠. 이번에는 Sex(성별) 변수를 활용해보겠습니다.

Sex 변수부터 살펴볼까요? train 데이터의 Sex 컬럼은 'male'과 'female', 두 가지 값을 가지고 있습니다. 문제는 이 데이터를 그대로 머신러닝 모델에 입력할 수 없다는 점입니다. 대부분의 머신러닝 알고리즘은 수치 데이터만 처리할 수 있기 때문이죠. 문자열인 'male'과 'female'을 컴퓨터가 이해할 수 있는 숫자로 변환해야 합니다. 이러한 변환 과정을 '인코딩(encoding)'이라고 부릅니다.

가장 간단한 방법은 'female'을 0으로, 'male'을 1로 변환하는 것입니다. 이를 '레이블 인코딩(Label Encoding)'이라고 합니다. 또는 'male'과 'female'을 각각의 새로운 피처로 만드는 '원-핫 인코딩(One-Hot Encoding)' 방법도 있습니다. 원-핫 인코딩을 사용하면 Sex_male과 Sex_female이라는 두 개의 컬럼이 생성되고, 해당하는 성별의 컬럼에만 1을, 나머지 컬럼에는 0을 설정합니다. 두 방법 중 어떤 것을 선택해야 할까요?

이 경우에는 레이블 인코딩을 사용하겠습니다. Sex는 두 가지 값만 가지는 이진 변수입니다. 원-핫 인코딩을 하면 두 개의 컬럼이 생성되는데, 이는 불필요한 중복입니다. 하나의 컬럼으로도 충분히 정보를 표현할 수 있기 때문입니다.

이제 Sex 변수를 레이블 인코딩해보겠습니다.

📝 **프롬프트**

```
Sex 피처를 인코딩해서 Gender라는 새로운 피처를 만들어줘
'male'은 0으로, 'female'은 1로 인코딩해줘
```

💻 **코드 & 실행결과**

```{code-cell}
# train 데이터에 Gender 피처 생성
train['Gender'] = train['Sex'].map({'male': 0, 'female': 1})

# test 데이터에도 Gender 피처 생성
test['Gender'] = test['Sex'].map({'male': 0, 'female': 1})
```

레이블 인코딩을 수행하는 방법은 여러 가지가 있습니다. 여기서는 pandas의 map 함수를 사용했습니다. map 함수는 딕셔너리를 입력으로 받아 키(key)에 해당하는 값을 값(value)으로 변환합니다. Sex 피처의 값이 'male'이면 0으로, 'female'이면 1로 변환하여 Gender라는 새로운 피처에 저장했습니다. 
train 데이터와 test 데이터 모두에 동일한 인코딩을 적용한 것에 주목해주세요. 머신러닝 프로젝트에서는 train 데이터와 test 데이터에 항상 동일한 전처리를 적용해야 합니다.



그럼 변환이 제대로 되었는지 확인해보겠습니다. 변환 결과를 더 쉽게 확인할 수 있도록 필요한 열만 선택해서 출력하겠습니다. 

📝 **프롬프트**

```
train에서 'PassengerId', 'Survived', 'Sex', 'Gender' 컬럼만 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
train[['PassengerId', 'Survived', 'Sex', 'Gender']].head()
```

Sex 피처의 'male', 'female' 값들이 Gender 피처에서는 0과 1로 잘 변환된 것을 확인할 수 있습니다.



## 모델 학습과 예측

이제 1.3절의 "Model Training" 과정을 따라 Random Forest 모델을 학습하겠습니다. Gender 피처를 포함하는 방법은 두 가지가 있습니다.

첫 번째는 기존의 inc_fts 리스트에 'Gender'를 추가하는 방법입니다:
inc_fts += ['Gender']  # 기존 리스트에 'Gender' 추가

또는 처음부터 모든 피처를 포함하여 리스트를 만드는 방법도 있습니다:
inc_fts = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Gender']

💻 **코드 & 실행결과**

```{code-cell}
inc_fts = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
inc_fts += ['Gender']  # 기존 리스트에 'Gender' 추가

print(inc_fts)
```
어느 방법을 사용하든 결과는 동일합니다. 저는 첫 번째 방법으로 하겠습니다. 나머지 학습 과정은 이전과 동일하게 진행하면 됩니다.



💻 **코드 & 실행결과**

```{code-cell}
X = train[inc_fts]     # 선택한 정보들
y = train['Survived']  # 실제 생존 여부
X_test = test[inc_fts] # 예측해야 할 데이터의 정보들

print(X.shape, y.shape, X_test.shape) # 행과 열 출력 
```

X.shape의 결과가 (891, 6)으로 출력되는 것을 확인할 수 있습니다. 이는 891개의 데이터에 6개의 피처(Pclass, Age, SibSp, Parch, Fare, Gender)가 있다는 의미입니다. 마찬가지로 X_test.shape도 (418, 6)으로 출력됩니다. test 데이터의 418개 데이터에도 동일한 6개의 피처가 있다는 뜻이죠.



💻 **코드 & 실행결과**

```{code-cell}
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_valid.shape
```



💻 **코드 & 실행결과**

```{code-cell}
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```



💻 **코드 & 실행결과**

```{code-cell}
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation Score: {accuracy:.5f}")
```

모델 학습 결과, Validation Score가 0.82123으로 나왔습니다. 이는 챕터 1에서 수치형 변수만 사용했을 때의 점수인 0.73184보다 크게 향상된 결과입니다. Gender 피처 하나를 추가했을 뿐인데 정확도가 많이 상승했네요. 이는 앞서 살펴본 것처럼 성별이 생존 여부와 매우 강한 관련이 있다는 것을 다시 한 번 확인시켜주는 결과입니다.



💻 **코드 & 실행결과**

```{code-cell}
y_test_pred = model.predict(X_test)

submission['Survived'] = y_test_pred
submission.to_csv('titanic_pred.csv', index=False)
submission.head()
```

실제 test 데이터에 대한 예측 결과를 캐글에 제출했더니 0.73444의 점수를 얻었습니다. 이는 418명의 test 데이터 중 307명의 생존 여부를 맞혔다는 의미입니다. 챕터 1에서 수치형 변수만 사용했을 때의 점수가 0.63397(265명)이었던 것과 비교하면, Gender 피처를 추가함으로써 42명을 더 정확하게 예측할 수 있게 되었네요.



지금까지의 결과를 표로 정리하면 다음과 같습니다:

| 버전 | 피쳐 개수 | Val. Score | Public Score | 맞은 사람 수 |          설명           |
| :--: | :-------: | :--------: | :----------: | :----------: | :---------------------: |
| 1.3  |     5     |  0.73184   |   0.63397    |    265명     | 5개의 numeric 피쳐 사용 |
| 2.1  |     6     |  0.82123   |   0.73444    |    307명     |    Gender 피쳐 추가     |



