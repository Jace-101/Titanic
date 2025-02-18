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

for df in [train, test]:
    df['Gender'] = df['Sex'].map({'male': 0, 'female': 1})
```

```{code-cell}
:tags: ["hide-input"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_predict(train, test):
    # 데이터 준비
    X = train[inc_fts]     # 선택한 특성들
    y = train['Survived']  # 생존 여부
    X_test = test[inc_fts] # 예측해야 할 데이터의 정보들

    # 학습/검증 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 성능 평가
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Score: {accuracy:.5f}")

    # 테스트 데이터 예측 및 저장
    y_test_pred = model.predict(X_test)
    submission['Survived'] = y_test_pred
    submission.to_csv('titanic_pred.csv', index=False)
```

<br>

# 2. 승선 항구(Embarked)가 비어있는 두 승객엔 어떤 비밀이 있을까?

성별 정보의 추가로 우리의 예측 모델은 큰 진전을 이루었습니다. 하지만 타이타닉호의 생존과 죽음을 가른 것이 단순히 성별뿐이었을까요? 이번에는 승객들의 승선 항구를 나타내는 'Embarked' 변수를 살펴보겠습니다. 과연 어느 항구에서 승선한 승객들의 생존 확률이 더 높았을까요?



## Embarked 피쳐 개요

먼저 Embarked 변수에 어떤 값들이 있는지 확인해보겠습니다.

📝 **프롬프트**

```
Embarked 피쳐에 어떤 값들이 있는지 보여줘
```

💻 **코드 & 실행결과**

```{code-cell}
print(train['Embarked'].unique())
```

결과를 보면 네 가지 값이 있습니다:

- 'S': Southampton(영국)
- 'C': Cherbourg(프랑스)
- 'Q': Queenstown(아일랜드)
- nan: 결측치(Not a Number)

```{figure} ./images/22-1.png
---
---
```

타이타닉호는 영국 Southampton에서 출발하여 프랑스 Cherbourg, 아일랜드 Queenstown을 거쳐 뉴욕으로 향했습니다. 각 항구는 서로 다른 이야기를 품고 있었습니다:

- Southampton: 타이타닉호의 첫 출발지이자 가장 많은 승객이 탑승한 항구
- Cherbourg: 프랑스 상류층이 선호하던 항구. 유럽 대륙의 귀족들이 주로 이용
- Queenstown: 많은 아일랜드 이민자들이 '아메리칸 드림'을 꿈꾸며 출발했던 곳

흥미로운 점은 일부 승객의 승선 항구 정보가 누락되어 있다는 것입니다. 이들은 어떤 사연을 가진 승객들일까요?



## 출발지별 생존율 분석

먼저 각 항구에서 승선한 승객들의 생존 확률이 어떻게 달랐는지부터 살펴보겠습니다.

📝 **프롬프트**

```
출발지(Embarked)별 승객 수와 생존율을 계산해줘
```

💻 **코드 & 실행결과**

```{code-cell}
embarked_stats = train.groupby('Embarked')['Survived'].agg(['count', 'mean'])
embarked_stats
```

분석 결과가 보여주는 놀라운 패턴들을 살펴보겠습니다:

1. 승객 분포:
   - Southampton(S)에서 가장 많은 644명이 승선했습니다
   - Cherbourg(C)에서 168명이 승선했습니다
   - Queenstown(Q)에서 가장 적은 77명이 승선했습니다
2. 생존율:
   - Cherbourg 승선객의 생존율이 55.4%로 가장 높습니다
   - Queenstown 승선객의 생존율은 39.0%입니다
   - Southampton 승선객의 생존율이 33.7%로 가장 낮습니다

특히 주목할 만한 점은 Cherbourg에서 승선한 승객들의 생존율이 다른 항구 승선객들에 비해 현저히 높다는 것입니다. 이는 Cherbourg가 프랑스의 부유한 항구도시였고, 이곳에서 승선한 승객들 중 상당수가 1등실을 이용했기 때문으로 보입니다.



## 결측치의 수 확인

먼저 Embarked 피쳐에 정확히 몇 개의 결측치가 있는지 확인해보겠습니다. 전체적인 결측치 현황을 파악하는 것은 데이터 전처리의 첫 단계입니다.  train 데이터와 test 데이터 모두 확인해보겠습니다.

📝 **프롬프트**

```
train과 test의 Embarked 결측치 개수를 알려줘
```

💻 **코드 & 실행결과**

```{code-cell}
print(f"train 데이터의 Embarked 결측치 개수: {train['Embarked'].isnull().sum()}")
print(f"test 데이터의 Embarked 결측치 개수: {test['Embarked'].isnull().sum()}")
```

결과를 보면 train 데이터에는 2개, test 데이터에는 0개의 결측치가 있음을 알 수 있습니다. 즉, 승선 항구 정보가 누락된 승객은 train 데이터에만 2명 있으며, test 데이터에는 없습니다. train 데이터의 결측치는 전체 891명의 승객 중 0.2% 정도로 매우 적은 비율이지만, 정확한 예측을 위해서는 이 결측치들도 적절히 처리하는게 좋습니다.



## 결측치의 비밀 추적하기

이제 승선 항구 정보가 누락된 승객들의 이야기를 추적해보겠습니다. 이들은 누구이며, 어떤 사연이 있었을까요?

📝 **프롬프트**

```
Embarked가 결측치인 승객들의 정보를 보여줘
```

💻 **코드 & 실행결과**

```{code-cell}
train[train['Embarked'].isnull()]
```

발견된 단서들을 정리해보면:

1. 같은 특성:
   - 두 승객 모두 1등실 여성 승객입니다
   - 지불한 운임이 모두 80파운드로 동일합니다
   - 같은 티켓으로 같은 객실을 사용했습니다
   - 두 승객 모두 생존했습니다
2. 다른 특성:
   - 성(lastname)이 다릅니다
   - 나이는 38세와 62세로 24살 차이가 납니다
   - 둘 다 가족(SibSp=0, Parch=0)과 함께 타지 않았습니다

:::{admonition} 생각해 보기 

1912년, 혼자 여행하기 어려웠던 시대에 이 두 여성은 어떤 관계였을까요? 다음과 같은 가능성들을 생각해볼 수 있습니다:

- 부유한 여성과 그의 여행 동반자(companion)
- 귀부인과 그의 가정교사 또는 개인비서
- 친구 사이였을까요? 아니면 다른 관계였을까요?

여러분은 어떤 관계였을 것 같나요?

:::

이러한 추리는 단순한 재미를 넘어 실제 데이터 분석에도 도움이 됩니다. 두 승객이 동행이었다면, 당연히 같은 항구에서 승선했을 것이기 때문입니다. 앞으로의 결측치 처리에서 이 점을 고려해야 할 것입니다.



## 1등석 승객들의 승선 항구 분석

두 승객이 모두 1등석을 이용했다는 점은 중요한 단서가 됩니다. 1등석 승객들은 주로 어느 항구에서 승선했을까요?

📝 **프롬프트**
```
Pclass가 1인 승객들의 Embarked 값이 몇 명인지 세어 줘
```

💻 **코드 & 실행결과**
```{code-cell}
train[train['Pclass'] == 1]['Embarked'].value_counts()
```

결과를 보면 1등석 승객들의 승선 패턴이 뚜렷하게 나타납니다:

- Southampton(S)에서 127명이 승선했습니다
- Cherbourg(C)에서 85명이 승선했습니다
- Queenstown(Q)에서는 단 2명만이 승선했습니다

이 분포는 결측치 승객들의 승선 항구를 추정하는데 중요한 힌트를 제공합니다. Queenstown은 거의 가능성이 없어 보이며, Southampton과 Cherbourg 중 하나일 것입니다.



## 결측치 처리하기

이전 분석을 통해 우리는 중요한 단서들을 발견했습니다:

- 두 승객은 같은 티켓으로 1등실을 이용했습니다
- 1등석 승객들의 다수(127명)가 Southampton에서 승선했습니다
- Queenstown에서는 1등석 승객이 단 2명뿐이었습니다

이러한 패턴을 고려할 때, 이 두 승객도 Southampton에서 승선했을 가능성이 높아 보입니다. 이에 따라 결측치를 'S'로 채워보겠습니다.

📝 **프롬프트**
```
1. Embarked 값이 비어 있는 승객들의 값을 'S'로 채워 줘
2. PassengerId가 62 또는 830인 승객들의 정보를 보여 줘
```

💻 **코드 & 실행결과**
```{code-cell}
train['Embarked'] = train['Embarked'].fillna('S')
train[train['PassengerId'].isin([62, 830])]
```

실행 결과를 보면 두 승객의 Embarked 값이 NaN에서 'S'로 변경된 것을 확인할 수 있습니다. 이로써 train 데이터의 Embarked 결측치 처리가 완료되었습니다. 이제 다음 단계로 넘어가서, 이 정보를 머신러닝 모델이 이해할 수 있는 형태로 변환해보도록 하겠습니다.



## 인코딩 방식 비교

승선 항구 정보를 숫자로 변환하는 방법에는 크게 두 가지가 있습니다. 앞서 성별 데이터를 처리할 때 사용했던 레이블 인코딩과, 이번에 사용할 원-핫 인코딩입니다. 위 다이어그램은 두 방식의 차이를 보여줍니다.

레이블 인코딩은 각 범주에 숫자를 할당하는 방식입니다:

- Southampton = 0
- Cherbourg = 1
- Queenstown = 2

하지만 이 방식은 중요한 문제가 있습니다. 숫자의 크기가 마치 항구 간의 순위나 중요도를 나타내는 것처럼 보일 수 있기 때문입니다. Queenstown(2)이 Southampton(0)보다 "더 좋은" 항구인 것처럼 해석될 수 있죠.

반면 원-핫 인코딩은 각 항구를 별도의 컬럼으로 만듭니다:

- Embarked_S: Southampton에서 승선했으면 1, 아니면 0
- Embarked_C: Cherbourg에서 승선했으면 1, 아니면 0
- Embarked_Q: Queenstown에서 승선했으면 1, 아니면 0

이렇게 하면 각 항구가 독립적으로 처리되어, 불필요한 순서 관계가 생기는 것을 방지할 수 있습니다.

📝 **프롬프트**
```
train과 test의 Embarked 피쳐를 One-Hot Encoding으로 추가해줘
```

💻 **코드 & 실행결과**
```{code-cell}
embarked_dummies_train = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_dummies_train], axis=1)

embarked_dummies_test = pd.get_dummies(test['Embarked'], prefix='Embarked')
test = pd.concat([test, embarked_dummies_test], axis=1)

train.head()
```

이제 'Embarked_S', 'Embarked_C', 'Embarked_Q' 세 개의 새로운 컬럼이 생성되었고, 각 승객은 자신이 승선한 항구에 해당하는 컬럼에만 1의 값을 가지게 됩니다. 이렇게 하면 항구 간에 크기 관계가 생기는 것을 방지할 수 있습니다.



:::{admonition} One-Hot Encoding의 장단점

**장점**:

- 범주형 변수들 사이에 크기 관계가 생기는 것을 방지할 수 있습니다
- 각 범주가 독립적으로 처리되어 모델이 더 정확한 패턴을 학습할 수 있습니다

**단점**:
- 범주의 종류가 많으면 생성되는 피쳐의 수도 많아집니다
- 메모리 사용량이 증가하고 학습 시간이 길어질 수 있습니다
- 과적합의 위험이 증가할 수 있습니다 
:::



## 모델 학습과 예측

이제 원-핫 인코딩으로 생성된 새로운 피쳐들을 사용하여 모델을 학습시켜 보겠습니다. 앞서 사용한 피쳐들(Pclass, Age, SibSp, Parch, Fare, Gender)에 승선 항구 관련 피쳐들을 추가하여 예측의 정확도를 높여보겠습니다.

📝 **프롬프트**
```
승선 항구 관련 피쳐들(Embarked_C, Embarked_Q, Embarked_S)을 inc_fts에 추가해줘
```

💻 **코드 & 실행결과**
```{code-cell}
inc_fts = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
inc_fts += ['Gender']
inc_fts += ['Embarked_C', 'Embarked_Q', 'Embarked_S']

print(inc_fts)
```

이제 총 9개의 피쳐를 사용하게 됩니다:
- 처음에 사용했던 5개의 수치형 피쳐
- 성별을 나타내는 Gender 피쳐
- 승선 항구를 나타내는 3개의 피쳐(Embarked_C, Embarked_Q, Embarked_S)

모델의 학습과 예측은 이제 train_and_predict(train, test) 함수 하나로 진행됩니다. 1.3 섹션에서 사용했던 모든 모델 학습 코드가 이 함수 안에 포함되어 있어서, 매번 긴 코드를 반복해서 입력할 필요가 없어졌습니다. 마치 요리 책에서 자주 사용하는 기본 양념을 미리 섞어두는 것처럼, 자주 사용하는 코드를 하나의 함수로 정리한 것이죠.

💻 **코드 & 실행결과**
```{code-cell}
train_and_predict(train, test)
```

이렇게 간단한 명령어 하나로 모델 학습부터 예측까지 모든 과정이 자동으로 실행됩니다. 이제 우리는 피쳐를 추가하거나 수정할 때 inc_fts 리스트만 변경하면 되니, 더욱 효율적으로 실험을 진행할 수 있게 되었습니다.



이렇게 승선 항구 정보를 추가하여 학습한 모델의 성능을 보면:
- Validation Score: 0.82123 (이전과 동일)
- Public Score: 0.74880 (이전 0.73444에서 상승)
- 418명의 test 데이터 중 약 313명의 생존 여부를 맞혔습니다 (이전보다 6명 증가)

지금까지의 결과를 표로 정리하면 다음과 같습니다:

| 버전 | 피쳐 개수 | Val. Score | Public Score | 맞은 사람 수 |          설명           |
| :--: | :-------: | :--------: | :----------: | :----------: | :---------------------: |
| 1.3  |    5개    |  0.73184   |   0.63397    |    265명     | 5개의 numeric 피쳐 사용 |
| 2.1  |    6개    |  0.82123   |   0.73444    |    307명     |    Gender 피쳐 추가     |
| 2.2  |    9개    |  0.82123   |   0.74880    |    313명     |   승선 항구 정보 추가   |

이 결과는 매우 흥미로운 시사점을 제공합니다:

1. 성별(Gender)의 강력한 영향력

2. 승선 항구의 보완적 역할
   - 3개의 피쳐가 추가되었지만 6명의 예측만 개선되었습니다
   - 하지만 이는 각 항구의 고유한 특성이 생존에 영향을 미쳤음을 의미합니다
   - 특히 Cherbourg 승객들의 높은 생존율(55.4%)이 예측 정확도 향상에 크게 기여했습니다
   - 이는 Cherbourg에서 승선한 승객들 중 1등실 승객의 비율이 높았기 때문입니다
   - 앞서 분석에서 보았듯이, 전체 168명의 Cherbourg 승객 중 무려 85명(약 51%)이 1등실을 이용했습니다


지금까지 승선 항구별 특성을 살펴보았는데, 특히 Cherbourg 승객들의 높은 생존율은 이들이 지불한 요금과도 관련이 있을 것 같습니다. 다음 섹션에서는 요금(Fare) 데이터를 자세히 분석하면서, 이 변수가 생존율에 미치는 영향을 살펴보도록 하겠습니다.



:::{admonition} 프롬프트 실험하기

AI에게 분석을 요청할 때는 다양한 방식으로 프롬프트를 작성해볼 수 있습니다:

1. 기본적인 분석 요청:
   ```
   Embarked 컬럼의 값들을 보여줘
   ```

2. 구체적인 분석 요청:
   ```
   각 항구별로 1등석, 2등석, 3등석 승객의 수와 비율을 계산해줘
   ```

3. 시각화 요청:
   ```
   항구별 승객 수와 생존율을 막대그래프로 보여줘
   ```

4. 복합적인 분석 요청:
   ```
   각 항구별로 성별 분포와 평균 요금을 함께 보여주고, 
   이것이 생존율과 어떤 관계가 있는지 분석해줘
   ```

같은 데이터도 다양한 각도에서 분석할 수 있습니다. 여러분만의 프롬프트로 새로운 인사이트를 발견해보세요!
:::



:::{admonition} 데이터 속 숨은 이야기
:class: seealso

우리가 분석한 결측치 승객들의 실제 이야기를 찾아보았습니다!

[Encyclopedia Titanica](https://www.encyclopedia-titanica.org)에서 이 두 승객에 대한 흥미로운 기록을 발견했습니다. 데이터에서 우리가 추측했던 것처럼, 이들은 실제로 함께 여행했던 동행이었습니다.

```{figure} images/22-2.png
---
---
```

62세의 Stone 부인(사진)과 38세의 Miss Icard는 고용주와 가정부 관계였습니다. 둘은 Southampton에서 승선했고, 다행히 모두 생존했습니다. 특히 감동적인 것은 Stone 부인이 사망 시 Miss Icard에게 상당한 금액의 유산과 개인 소지품을 남겼다는 기록입니다. 24년의 나이 차이를 넘어, 고용주와 가정부로 시작된 두 사람의 관계가 깊은 신뢰와 우정으로 발전했던 것 같네요.

이렇게 차가운 숫자로 이루어진 데이터 속에도 따뜻한 인간 드라마가 숨어있습니다. 우리가 분석한 결측치 처리가 단순한 데이터 작업이 아닌, 실제 존재했던 사람들의 이야기라는 점이 특별하게 다가옵니다.

더 자세한 내용이 궁금하시다면:
- [Martha Evelyn Stone의 이야기](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html)

- [Amelia Icard의 이야기](https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html)

:::

