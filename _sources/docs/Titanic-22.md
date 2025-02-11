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



# 2. 탑승 항구(Embarked)가 비어있는 두 승객엔 어떤 비밀이 있을까?

💻 **준비 코드**

```{code-cell}
:tags: ["hide-input"]

import pandas as pd

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
submission = pd.read_csv('./gender_submission.csv')

for df in [train, test]:
    df['Gender'] = df['Sex'].map({'male': 0, 'female': 1})
```



## Embarked 피쳐 개요

📝 **프롬프트**

```
Embarked 피쳐에 어떤 값들이 있는지 보여줘
```

💻 **코드 & 실행결과**

```{code-cell}
train['Embarked'].unique()
```

위 코드는 Embarked 피쳐에 존재하는 고유한 값들을 확인하는 작업을 수행합니다. unique() 함수는 데이터프레임의 특정 컬럼에서 중복을 제외한 고유한 값들을 배열로 반환합니다.

결과를 보면 네 가지 값이 있음을 알 수 있습니다:

- 'S': Southampton(영국)
- 'C': Cherbourg(프랑스)
- 'Q': Queenstown(아일랜드)
- nan: 결측치(Not a Number)를 의미

이 중 nan이 있다는 것은 일부 승객의 탑승 항구 정보가 누락되어 있다는 뜻입니다. 이러한 결측치는 앞으로 자세히 분석하고 적절히 처리해야 할 대상입니다.



## 타이타닉호의 항로

타이타닉호는 영국 Southampton에서 출발하여 프랑스 Cherbourg, 아일랜드 Queenstown을 거쳐 미국 뉴욕으로 향했습니다. 각 항구에서 승객들이 탑승했으며, 이는 당시 대서양 횡단의 일반적인 항로였습니다.

```{figure} ./images/22-1.png
---
---
```



## 출발지별 생존율 분석

각 출발지별 생존율을 분석하면 어떤 항구에서 탑승한 승객들의 생존 확률이 더 높았는지 알 수 있을 것입니다. 이러한 정보는 결측치를 적절한 값으로 대체하는 데에도 도움이 될 수 있습니다. 먼저 출발지별 승객 수와 생존율을 계산해보겠습니다.

📝 **프롬프트**

```
출발지(Embarked)별 승객 수와 생존율을 계산해줘
```

💻 **코드 & 실행결과**

```{code-cell}
results = train.groupby('Embarked')['Survived'].agg(['count', 'mean'])
results
```

이 코드는 각 출발지별 승객 수와 생존율을 계산합니다. 구체적으로:

1. `groupby('Embarked')`로 출발지를 기준으로 데이터를 그룹화합니다.

2. `['Survived']`로 생존 여부 컬럼을 선택합니다.

3. ```
   agg(['count', 'mean'])
   ```

   로 각 그룹에 대해 두 가지 계산을 수행합니다:

   - 'count': 각 출발지의 승객 수를 세어줍니다.
   - 'mean': Survived 컬럼(0 또는 1)의 평균을 계산하여 생존율을 구합니다.

분석 결과가 보여주는 흥미로운 패턴들을 살펴보겠습니다:

1. 승객 분포:
   - Southampton(S)에서 가장 많은 644명이 탑승했습니다.
   - Cherbourg(C)에서 168명이 탑승했습니다.
   - Queenstown(Q)에서 가장 적은 77명이 탑승했습니다.
2. 생존율:
   - Cherbourg 탑승객의 생존율이 55.4%로 가장 높습니다.
   - Queenstown 탑승객의 생존율은 39.0%입니다.
   - Southampton 탑승객의 생존율이 33.7%로 가장 낮습니다.

특히 주목할 만한 점은 Cherbourg에서 탑승한 승객들의 생존율이 다른 항구 탑승객들에 비해 현저히 높다는 것입니다. 이는 Cherbourg에서 탑승한 승객들 중 상대적으로 더 많은 비율이 1등실 승객이었거나, 여성 승객의 비율이 더 높았을 가능성을 시사합니다. 이러한 정보는 앞으로 결측치를 처리할 때 중요한 참고사항이 될 수 있습니다.



## 결측치의 수 확인

먼저 Embarked 피쳐에 정확히 몇 개의 결측치가 있는지 확인해보겠습니다. 전체적인 결측치 현황을 파악하는 것은 데이터 전처리의 첫 단계입니다.  train 데이터와 test 데이터 모두 확인해보겠습니다.

📝 **프롬프트**

```
train과 test 데이터의 Embarked 컬럼 결측치 개수를 계산해줘
```

💻 **코드 & 실행결과**

```
print(train['Embarked'].isnull().sum())
print(test['Embarked'].isnull().sum())
```

이 코드는 Embarked 컬럼의 결측치 개수를 계산합니다:

1. `isnull()`은 각 값이 결측치인지 아닌지를 True/False로 반환합니다.
2. `sum()`은 True를 1로 계산하여 결측치의 총 개수를 반환합니다.

결과를 보면 train 데이터에는 2개, test 데이터에는 0개의 결측치가 있음을 알 수 있습니다. 즉, 탑승 항구 정보가 누락된 승객은 train 데이터에만 2명 있으며, test 데이터에는 없습니다. train 데이터의 결측치는 전체 891명의 승객 중 0.2% 정도로 매우 적은 비율이지만, 정확한 예측을 위해서는 이 결측치들도 적절히 처리하는게 좋습니다.



## 결측치 승객의 관계 추리

이 두 승객의 관계는 무엇이었을까요? 1912년이라는 시대 상황과 발견된 단서들을 바탕으로 추리해보면 재미있는 이야기가 그려집니다.

📝 **프롬프트**

```
Embarked가 결측치인 승객들의 정보를 보여줘
```

💻 **코드 & 실행결과**

```
train[train['Embarked'].isnull()]
```

이 코드는 Embarked 값이 누락된 승객들의 정보를 조회합니다:

1. `train['Embarked'].isnull()`로 Embarked가 결측치인 행을 True로 표시합니다.
2. 이 조건을 train 데이터프레임의 인덱싱에 사용하여 해당하는 행들만 선택합니다.



발견된 단서들을 정리해보면:

1. 같은 특성
   - 두 승객 모두 1등실 여성 승객입니다.
   - 지불한 운임이 모두 80파운드로 동일합니다.
   - 같은 티켓으로 같은 객실을 사용했습니다.
   - 두 승객 모두 생존했습니다.
2. 다른 특성
   - 성(lastname)이 다릅니다.
   - 나이는 38세와 62세로 24살 차이가 납니다.
   - 둘 다 가족(SibSp=0, Parch=0)과 함께 타지 않았습니다.

이 단서들을 바탕으로 1912년의 상황을 상상해볼까요? 혼자 여행하는 여성이 드물었던 당시, 같은 객실을 쓰는 여성 두 명이 있습니다. 가족은 아니지만 동행이었고, 나이 차이가 24살 납니다.

:::{admonition} 생각해보기 
:class: tip 

이 두 여성의 관계는 무엇이었을까요? 1900년대 초반, 혼자 여행하기 어려웠던 시대에 왜 이 두 여성은 함께 있었을까요? 다음과 같은 가능성들을 생각해볼 수 있습니다:

- 부유한 여성과 그의 여행 동반자(companion)
- 귀부인과 그의 가정교사 또는 개인비서
- 친구 사이였을까요? 아니면 다른 관계였을까요?

여러분은 어떤 관계였을 것 같나요? 
:::

이러한 추리는 단순한 재미를 넘어 실제 데이터 분석에도 도움이 됩니다. 두 승객이 동행이었다면, 당연히 같은 항구에서 탑승했을 것이기 때문입니다. 앞으로의 결측치 처리에서 이 점을 고려해야 할 것입니다.

## 1등석 승객들의 탑승항구 분석

두 승객이 모두 1등석을 이용했다는 점이 중요한 단서가 될 수 있습니다. 1등석 승객들은 주로 어느 항구에서 탑승했는지 살펴보면 결측치를 채우는데 도움이 될 것입니다.

📝 **프롬프트**
```
1등석(Pclass=1) 승객들의 탑승항구 분포를 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
train[train['Pclass'] == 1]['Embarked'].value_counts()
```

이 코드는 1등석 승객들의 탑승항구 분포를 계산합니다:
1. `train['Pclass'] == 1`로 1등석 승객만 선택합니다.
2. `['Embarked']`로 탑승항구 정보를 선택합니다.
3. `value_counts()`로 각 항구별 승객 수를 계산합니다.

결과를 보면 1등석 승객들의 탑승 패턴이 뚜렷하게 나타납니다:
- Southampton(S)에서 127명이 탑승했습니다.
- Cherbourg(C)에서 85명이 탑승했습니다.
- Queenstown(Q)에서는 단 2명만이 탑승했습니다.
- 그리고 우리가 찾던 2명의 승객이 결측치입니다.

이 분포는 매우 의미 있는 정보를 제공합니다. Queenstown은 1등석 승객이 거의 없었다는 점에서 우리가 찾는 두 승객의 탑승항구일 가능성이 매우 낮습니다. 결측치는 Southampton과 Cherbourg 중 한 곳일 가능성이 높아 보입니다. 추가 분석을 통해 이 두 항구 중 어느 곳이었는지 좀 더 구체적으로 알아보아야 할 것 같습니다.



## 결측치 값 채우기

이제 결정한 대로 결측치를 'S'로 채워보겠습니다.

📝 **프롬프트**
```
1. Embarked가 결측치인 승객들의 값을 'S'로 바꿔줘
2. 해당 승객들(PassengerId가 62 또는 830)의 정보를 다시 출력해줘
```

💻 **코드 & 실행결과**
```{code-cell}
train.loc[train['Embarked'].isnull(), 'Embarked'] = 'S'
train[train['PassengerId'].isin([62,830])]
```

이 코드는 두 가지 작업을 수행합니다:

1. 첫 번째 줄은 결측치를 'S'로 변경합니다:
   - `train['Embarked'].isnull()`로 Embarked가 결측치인 행을 찾습니다.
   - `loc`을 사용해 해당 행들의 'Embarked' 값을 'S'로 변경합니다.

2. 두 번째 줄은 변경된 결과를 확인합니다:
   - `PassengerId.isin([62,830])`으로 두 승객을 선택합니다.
   - 선택된 승객들의 모든 정보를 출력합니다.

결과를 보면 두 승객의 Embarked 값이 NaN에서 'S'로 변경된 것을 확인할 수 있습니다. 이로써 train 데이터의 Embarked 결측치 처리가 완료되었습니다.



## 탑승항구 인코딩

앞 섹션에서는 성별(Sex) 데이터를 처리할 때 레이블 인코딩을 사용했습니다. 이번에는 탑승항구(Embarked) 데이터를 처리하면서 또 다른 인코딩 방식인 원-핫 인코딩(One-Hot Encoding)을 사용해보겠습니다.

탑승항구는 'S', 'C', 'Q' 세 가지 값을 가집니다. 이러한 범주형 데이터를 숫자로 바꿀 때는 원-핫 인코딩이 더 적합할 수 있습니다. 예를 들어 'S'=1, 'C'=2, 'Q'=3과 같이 순서가 있는 숫자로 변환하면, 실제로는 없는 'C'가 'S'의 2배, 'Q'가 3배라는 관계가 생기기 때문입니다.

📝 **프롬프트**
```
1. train과 test 데이터의 Embarked 컬럼을 원-핫 인코딩으로 변환해줘
2. 생성된 더미변수들을 각각 train과 test 데이터에 추가해줘
```

💻 **코드 & 실행결과**
```{code-cell}
embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_dummies], axis=1)

embarked_dummies = pd.get_dummies(test['Embarked'], prefix='Embarked')
test = pd.concat([test, embarked_dummies], axis=1)
```

이 코드는 다음과 같은 작업을 수행합니다:

1. `pd.get_dummies()`로 원-핫 인코딩을 수행합니다:
   - 'S'는 [1, 0, 0]으로 변환됩니다.
   - 'C'는 [0, 1, 0]으로 변환됩니다.
   - 'Q'는 [0, 0, 1]으로 변환됩니다.
   - `prefix='Embarked'`로 새로 생성되는 컬럼들의 이름을 'Embarked_S', 'Embarked_C', 'Embarked_Q'로 지정합니다.

2. `pd.concat()`으로 원-핫 인코딩된 결과를 원본 데이터에 추가합니다:
   - `axis=1`은 열 방향으로 데이터를 이어붙인다는 의미입니다.
   - train과 test 데이터 모두에 동일한 방식으로 적용합니다.

이제 'Embarked_S', 'Embarked_C', 'Embarked_Q' 세 개의 새로운 컬럼이 생성되었고, 각 승객은 자신이 탑승한 항구에 해당하는 컬럼에만 1의 값을 가지게 됩니다. 이렇게 하면 항구 간에 크기 관계가 생기는 것을 방지할 수 있습니다.



:::{admonition} One-Hot Encoding의 장단점
:class: tip

**장점**:
- 범주형 변수들 사이에 크기 관계가 생기는 것을 방지할 수 있습니다.
- 각 범주가 독립적으로 처리되어 모델이 더 정확한 패턴을 학습할 수 있습니다.

**단점**:
- 범주의 종류가 많아지면 생성되는 피쳐의 수도 그만큼 많아집니다.
  - 예: 도시 이름이 100개라면 100개의 새로운 컬럼이 생성됩니다.
  - 예: 우편번호처럼 범주가 수천 개인 경우 사용하기 어렵습니다.
- 너무 많은 피쳐가 생성되면:
  - 데이터의 크기가 커져 메모리 사용량이 증가합니다.
  - 모델의 학습 시간이 길어집니다.
  - 과적합(overfitting)의 위험이 증가할 수 있습니다.

이런 단점 때문에, 범주의 종류가 많은 경우에는 다른 인코딩 방식(예: Label Encoding, Target Encoding 등)을 고려해야 합니다.
:::



## 모델 학습과 예측

이제 원-핫 인코딩으로 생성된 새로운 피쳐들을 사용하여 모델을 학습시켜 보겠습니다. 앞서 사용한 피쳐들(Pclass, Age, SibSp, Parch, Fare, Gender)에 탑승항구 관련 피쳐들을 추가하여 예측의 정확도를 높여보겠습니다.

📝 **프롬프트**
```
탑승항구 관련 피쳐들(Embarked_C, Embarked_Q, Embarked_S)을 inc_fts에 추가해줘
```

💻 **코드 & 실행결과**
```{code-cell}
inc_fts = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
inc_fts += ['Gender']
inc_fts += ['Embarked_C', 'Embarked_Q', 'Embarked_S']  # Embarked Encoding

print(inc_fts)
```

이 코드는 모델 학습에 사용할 피쳐 목록을 업데이트합니다:
1. 기존 피쳐 목록(Pclass, Age, SibSp, Parch, Fare, Gender)에
2. 원-핫 인코딩으로 생성된 세 개의 피쳐(Embarked_C, Embarked_Q, Embarked_S)를 추가합니다.

결과를 보면 이제 총 9개의 피쳐를 사용하게 됩니다:
- 수치형 피쳐 5개: Pclass, Age, SibSp, Parch, Fare
- 레이블 인코딩된 피쳐 1개: Gender
- 원-핫 인코딩된 피쳐 3개: Embarked_C, Embarked_Q, Embarked_S

이렇게 다양한 방식으로 인코딩된 피쳐들을 함께 사용하면, 모델이 더 많은 패턴을 학습할 수 있어 예측 성능이 향상될 것으로 기대됩니다.



지금까지의 결과를 표로 정리하면 다음과 같습니다:

| 버전 | 피쳐 개수 | 사용한 피쳐 | Val. Score | Public Score |              설명               |
| :--: | :-------: | :---------: | :--------: | :----------: | :-----------------------------: |
| 1.3  |    5개    | 수치형 5개  |  0.73184   |   0.63397    | Pclass, Age, SibSp, Parch, Fare |
| 2.1  |    6개    |  + Gender   |  0.82123   |   0.73444    |         성별 정보 추가          |
| 2.2  |    9개    | + Embarked  |  0.82123   |   0.74880    |       탑승항구 정보 추가        |

탑승항구 정보를 추가한 결과를 살펴보면:

1. Validation Score는 이전과 동일한 0.82123을 기록했습니다. 

2. 하지만 Public Score는 0.74880으로, 이전 버전(0.73444)보다 소폭 상승했습니다. 418명의 test 데이터 중 약 313명의 생존 여부를 맞혔다는 의미입니다. 이는 이전 버전보다 6명 더 정확하게 예측한 것입니다.

이러한 결과는 탑승항구 정보가 생존 예측에 어느 정도 도움이 되지만, 성별만큼 강력한 예측 변수는 아니라는 것을 보여줍니다. 실제로 2.1절에서 성별 정보를 추가했을 때는 점수가 크게 향상(0.63397 → 0.73444)되었지만, 이번에 탑승항구 정보를 추가했을 때는 상대적으로 작은 향상(0.73444 → 0.74880)만 있었습니다. 특히 피쳐의 개수는 6개에서 9개로 50% 증가했는데 비해, 성능 향상은 미미했다는 점이 주목할 만합니다.



:::{admonition} 데이터 속 숨은 이야기
:class: note

우리가 분석한 결측치 승객들의 실제 이야기를 찾아보았습니다!

[Encyclopedia Titanica](https://www.encyclopedia-titanica.org)에서 이 두 승객에 대한 흥미로운 기록을 발견했습니다. 데이터에서 우리가 추측했던 것처럼, 이들은 실제로 함께 여행했던 동행이었습니다.

```{figure} images/22-2.png
---
---
```

62세의 Stone 부인(사진)과 38세의 Miss Icard는 고용주와 가정부 관계였습니다. 둘은 Southampton에서 탑승했고, 다행히 모두 생존했습니다. 특히 감동적인 것은 Stone 부인이 사망 시 Miss Icard에게 상당한 금액의 유산과 개인 소지품을 남겼다는 기록입니다. 24년의 나이 차이를 넘어, 고용주와 가정부로 시작된 두 사람의 관계가 깊은 신뢰와 우정으로 발전했던 것 같네요.

이렇게 차가운 숫자로 이루어진 데이터 속에도 따뜻한 인간 드라마가 숨어있습니다. 우리가 분석한 결측치 처리가 단순한 데이터 작업이 아닌, 실제 존재했던 사람들의 이야기라는 점이 특별하게 다가옵니다.

더 자세한 내용이 궁금하시다면:
- [Martha Evelyn Stone의 이야기](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html)
- [Amelia Icard의 이야기](https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html)
:::

