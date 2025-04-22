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

# Section 2.1
for df in [train, test]:
    df['Gender'] = df['Sex'].map({'male': 0, 'female': 1})

# Section 2.2
train.loc[train['Embarked'].isnull(), 'Embarked'] = 'S'
train = pd.concat([train, pd.get_dummies(train['Embarked'], prefix='Embarked')], axis=1)
test = pd.concat([test, pd.get_dummies(test['Embarked'], prefix='Embarked')], axis=1)

# Section 2.3
median_fare = train[(train['Pclass'] == 3) & (train['Embarked'] == 'S')]['Fare'].median()
test['Fare'] = test['Fare'].fillna(median_fare)

# Section 2.5
age_by_pclass = train.groupby('Pclass')['Age'].median()
for pclass in [1, 2, 3]:
    train.loc[(train['Age'].isnull()) & (train['Pclass'] == pclass), 'Age'] = age_by_pclass[pclass]
for pclass in [1, 2, 3]:
    test.loc[(test['Age'].isnull()) & (test['Pclass'] == pclass), 'Age'] = age_by_pclass[pclass]

inc_fts = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
inc_fts += ['Gender']
inc_fts += ['Embarked_C', 'Embarked_Q', 'Embarked_S']
```

```{code-cell}
:tags: ["hide-input"]
!pip install catboost
```

<br>

# 1. 에이, 이름이 생존 여부에 영향을 줄 리 없잖아...

*데이터 과학 동아리 - 아홉 번째 모임*

---

프롬: 다안 선배! 지난 시간에는 숫자로 된 정보들을 이용해서 생존을 예측했잖아요. 근데 이름처럼 글자로 된 정보도 쓸 수 있어요?

다안: 물론이지, 프롬. 특히 이름에 들어 있는 'Mr', 'Miss', 'Dr' 같은 호칭은 단순한 말이 아니라 그 사람이 어떤 사람인지 힌트를 주는 정보야. 성별, 나이, 사회적 위치까지 짐작할 수 있거든.

코더블: 오, 그럼 이름 전체를 쓰는 게 아니라 그중에서 의미 있는 부분만 골라서 쓰는 거네요? 코드로 한번 뽑아볼게요.

```{code-cell}
# 이름에서 호칭 추출하기
for df in [train, test]:
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

train['Title'].value_counts()
````

프롬: 와, 정말 다양한 호칭이 있네요! 'Mr', 'Miss', 'Mrs' 말고도 'Don', 'Dr', 'Lady' 같은 건 처음 봤어요.

다안: 맞아. 타이타닉이 운항하던 1912년 당시엔 호칭만 봐도 나이, 성별, 신분을 알 수 있었어. 'Master'는 어린 남자아이, 'Countess'는 귀족 여성 같은 식이지.

프롬: 근데 이렇게 호칭이 너무 많으면 분석하기 힘들지 않을까요?

다안: 호칭이 다양하긴 하지만, 일단은 그대로 두고 어떤 특징이 있는지 살펴보자. 우선 호칭과 생존율 간의 관계를 확인해볼까?

코더블: 좋은 생각이에요. 각 호칭별로 생존율을 계산해볼게요.

```{code-cell}
# Title별 생존율 보기
train.groupby('Title')['Survived'].mean().sort_values(ascending=False)
```

프롬: 오! 'Miss'랑 'Mrs'는 생존율이 높고, 'Mr'는 낮아요. 'Master'도 생존율이 높네요. 정말 호칭만으로도 이런 차이가 있다니 신기해요!

다안: 그래. 당시에는 '여성과 아이 먼저'라는 구조 원칙이 있었잖아. 그래서 실제로 호칭만으로도 생존 가능성을 어느 정도 추측할 수 있었던 거야.

:::{note}
**호칭이 중요한 이유**

타이타닉에서는 구조 순서가 계급, 성별, 나이에 따라 정해졌어요.
- 'Mr': 성인 남성 → 생존율 낮음
- 'Miss', 'Mrs': 여성 → 생존율 높음
- 'Master': 어린 남자아이 → 약 57%의 생존율을 보였어요
- 특별한 호칭(Lady, Sir 등): 상류층 → 일반적으로 생존율 높음
:::

## CatBoost, 문자열도 OK!

코더블: 근데 'Title'은 글자잖아요? 모델에 바로 넣으면 안 되는 거 아닌가요?

다안: 일반적으로는 맞아. 대부분의 머신러닝 모델은 글자를 숫자로 바꿔줘야 해. 우리가 전에 쓴 RandomForest도 그렇고. 근데 오늘은 CatBoost라는 모델을 써볼 거야. 얘는 글자도 바로 이해할 수 있어.

프롬: 고양이랑 관련 있나요? 이름이 귀여워요!

다안: (웃음) 이름은 귀엽지만, 진짜 이름은 Categorical Boosting이야. 글자 형태 그대로, 즉 범주형 데이터를 처리하는 데 특화된 모델이지. 따로 인코딩하지 않아도 돼.

코더블: LabelEncoder나 OneHotEncoder 안 써도 된다니 정말 편하네요! 모델 성능도 좋다고 들었어요.

:::{admonition} CatBoost는 이런 점이 좋아요
:class: tip
- 문자열을 그대로 넣을 수 있어요
- 복잡한 인코딩 과정 없이도 높은 성능 가능
- 작은 데이터셋에서도 잘 작동해요
- 범주형 변수(카테고리 데이터)에 특화되어 있어요
:::

프롬: 직접 실행해보고 싶어요! 어떻게 사용하는 건가요?

다안: 먼저 CatBoost 라이브러리를 설치하고, 모델을 만들어볼게. 설치는 pip로 간단하게 할 수 있어.

코더블: 네, 저도 한번 해볼게요. 먼저 라이브러리부터 설치하고 함수를 만들어 볼게요.

```{code-cell}

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def CB_train_and_predict(train, test, cat_features=[], verbose=100):
    # 데이터 준비
    X = train[inc_fts]     # 선택한 특성들
    y = train['Survived']  # 생존 여부
    X_test = test[inc_fts] # 예측해야 할 데이터의 정보들

    # 학습/검증 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = CatBoostClassifier(random_state=42)
    model.fit(X_train, y_train, cat_features=cat_features, verbose=verbose)

    # 성능 평가
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Score: {accuracy:.5f}")

    # 테스트 데이터 예측 및 저장
    y_test_pred = model.predict(X_test)
    submission['Survived'] = y_test_pred
    submission.to_csv('titanic_pred.csv', index=False)
```

다안: 좋아. 이제 Title을 포함시켜서 모델을 학습시켜보자. 특히 cat_features 파라미터에 'Title'을 넣어줘야 해. 그럼 CatBoost가 알아서 Title을 범주형 변수로 처리해줄 거야.

코더블: 네, 그럼 Title을 features 목록에 추가하고 실행해 볼게요.

```{code-cell}
# Title을 features 목록에 추가
inc_fts += ['Title']

# CatBoost 모델 학습 및 예측
CB_train_and_predict(train, test, cat_features=['Title'])
```

프롬: 우와, 정확도가 0.826 이상이 나왔어요! 숫자만 썼을 때보다 좋아졌네요!

다안: 맞아. 이전에 랜덤 포레스트 모델에서는 약 0.82 정도였지? 호칭이라는 하나의 정보를 잘 활용했더니 성능이 향상됐어. 이게 바로 데이터 전처리의 힘이지.

코더블: 다른 문자 정보도 활용할 수 있지 않을까요? 예를 들면, 이름에서 성(Family name)을 추출해보는 건 어떨까요?

다안: 좋은 생각이야. 타이타닉에서는 같은 가족이 함께 탑승했을 가능성이 높으니까, 성을 통해 가족 관계를 파악할 수 있을지도 몰라.

프롬: 저도 궁금해요! 같은 가족이면 생존 여부도 비슷할 것 같아요.

코더블: 그럼 성(Lastname)을 추출해 볼게요.

```{code-cell}
# 이름에서 성(lastname) 추출하기
for df in [train, test]:
    df['Lastname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

print(train.Lastname.nunique(), test.Lastname.nunique())
```

프롬: 와, 훈련 데이터에 667개, 테스트 데이터에 352개의 다른 성이 있네요! 생각보다 많아요.

다안: 그렇지? 이제 이 정보도 모델에 추가해보자. 성도 호칭처럼 범주형 변수로 처리할 수 있어.

코더블: 네, Lastname도 features에 추가하고 다시 학습시켜 볼게요.

```{code-cell}
# Lastname을 features 목록에 추가
inc_fts += ['Lastname']

# CatBoost 모델 학습 및 예측
CB_train_and_predict(train, test, cat_features=['Title', 'Lastname'])
```

프롬: 아, 근데 정확도가 0.815로 조금 떨어졌어요. 왜 그런 걸까요?

다안: 좋은 질문이야. 모든 정보가 항상 도움이 되는 건 아니거든. 이 경우에는 성이 너무 다양해서 오히려 모델이 혼란스러워했을 수 있어. 우리 데이터셋에 비해 너무 많은 범주가 있으면 오버피팅이 일어날 수 있어.

코더블: 맞아요. 667개나 되는 다양한 성이 있는데, 각 성별로 데이터가 몇 개 없으니까 패턴을 찾기 어려웠을 거예요. 가족 단위로 그룹화하는 다른 방법을 찾아봐야 할 것 같아요.

프롬: 그럼 결국 호칭(Title)만 추가하는 게 가장 좋은 방법인가요?

다안: 지금까지 실험한 결과로는 그렇게 보이네. 하지만 이것도 중요한 발견이야. 모든 데이터가 항상 좋은 건 아니라는 걸 직접 확인했잖아. 데이터 과학에서는 이런 과정이 중요해.

프롬: 이름에서 호칭을 뽑아내서 모델 성능을 높였다니 정말 신기해요! 다음엔 어떤 정보를 활용해볼 수 있을까요?

다안: 좋은 질문이야. 다음에는 아직 우리가 사용하지 않은 'Cabin'이나 'Ticket' 정보에서도 패턴을 찾아볼 수 있을 거야. 또는 지금까지 배운 변수들을 더 효과적으로 조합해보는 것도 좋겠지.

코더블: 객실 번호는 선실의 위치와 관련이 있을 테니, 생존과도 관계가 있을 수 있겠네요!

---

:::{admonition} 직접 해보기
:class: tip
- 드문 호칭들을 묶어서 'Rare' 같은 새로운 카테고리로 만들어보세요. 성능이 어떻게 변할까요?
- Title과 다른 변수(예: Age, Pclass)를 조합해서 새로운 특성을 만들어보세요.
- Cabin 정보에서 선실의 구역(A, B, C 등)만 추출해서 모델에 활용해보세요.
:::
