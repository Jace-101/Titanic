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
    df.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'
    
median_fare = train[(train['Pclass'] == 3) & (train['Embarked'] == 'S')]['Fare'].median()
test['Fare'] = test['Fare'].fillna(median_fare)
```



# 3.1 이름에 숨겨진 생존 확률: Title(Mr, Mrs 등)의 비밀

결측치 처리와 승선 항구 분석을 마치고 이제 새로운 도전을 시작합니다. 승객들의 이름(Name) 컬럼에서 숨겨진 패턴을 찾아보려고 합니다. 단순한 문자열로 보이는 이름에서 과연 생존과 관련된 어떤 정보를 발견할 수 있을까요? 특히 Mr, Mrs, Miss와 같은 호칭(Title)이 당시의 사회적 지위나 결혼 여부를 나타내는 만큼, 이를 통해 생존율의 새로운 패턴을 찾아볼 수 있을 것 같습니다.

## 승객 이름의 구조 파악

먼저 이름 데이터가 어떤 구조로 되어있는지 살펴보겠습니다. 몇 가지 예시를 통해 패턴을 찾아보도록 하죠.

📝 **프롬프트**
```
train 데이터에서 Name 컬럼의 처음 5개 값을 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
train['Name'].head()
```

이름 데이터를 살펴보면 흥미로운 패턴이 보입니다:
- "Braund, Mr. Owen Harris"
- "Cumings, Mrs. John Bradley (Florence Briggs Thayer)"
- "Heikkinen, Miss. Laina"

각 이름은 세 부분으로 구성되어 있습니다:
1. 성(Last name): 쉼표(,) 앞의 부분
2. 호칭(Title): Mr., Mrs., Miss. 등
3. 이름(First name): 호칭 뒤의 부분

특히 호칭(Title)은 단순한 경칭이 아닌, 당시 승객의 사회적 신분이나 상황을 나타내는 중요한 정보였을 것 같습니다.

## 호칭(Title) 추출하기

이제 각 승객의 이름에서 호칭을 추출해보겠습니다. 호칭은 쉼표(,) 다음에 나오고 마침표(.)로 끝나는 부분입니다.

📝 **프롬프트**
```
train과 test 데이터의 Name 컬럼에서 호칭(Title)을 추출해서 새로운 컬럼으로 만들어줘
```

💻 **코드 & 실행결과**
```{code-cell}
for df in [train, test]:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')

train[['Name', 'Title']].head()
```

정규표현식을 사용하여 호칭을 성공적으로 추출했습니다. 이제 어떤 종류의 호칭들이 있는지 살펴보겠습니다.

📝 **프롬프트**
```
train 데이터에서 추출한 Title의 종류와 각각의 개수를 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
print("Train data titles:")
print(train['Title'].value_counts())
```

다양한 호칭들이 발견되었습니다:
- 일반적인 호칭: Mr, Mrs, Miss, Master
- 직업 관련: Dr, Rev(성직자), Col(대령), Major(소령)
- 귀족 호칭: Lady, Sir, Countess(백작부인), Don
- 기타: Mlle(프랑스어로 Miss), Mme(프랑스어로 Mrs)

이렇게 다양한 호칭들은 당시 승객들의 사회적 지위와 역할을 보여줍니다. 특히 주목할 만한 점은:
- 'Master'는 어린 남자아이를 지칭
- 'Miss'는 미혼 여성을 지칭
- 'Mrs'는 기혼 여성을 지칭
- 'Mr'는 성인 남성을 지칭

## 호칭별 생존율 분석

이제 각 호칭별로 생존율이 어떻게 다른지 살펴보겠습니다.

📝 **프롬프트**
```
Title별 승객 수와 생존율을 계산해서 내림차순으로 정렬해줘
```

💻 **코드 & 실행결과**
```{code-cell}
title_stats = train.groupby('Title')['Survived'].agg(['count', 'mean'])
title_stats = title_stats.sort_values('mean', ascending=False)
title_stats
```

호칭별 생존율을 보면 놀라운 패턴이 발견됩니다:
- 'Mme', 'Lady', 'Sir' 등 귀족 계급은 100% 생존
- 'Miss'와 'Mrs'는 약 70% 이상의 높은 생존율
- 'Master'(어린 남자아이)는 약 57%의 생존율
- 'Mr'(성인 남성)는 가장 낮은 16%의 생존율

이러한 결과는 타이타닉호 침몰 당시의 구조 우선순위를 반영합니다:
1. 귀족과 상류층 여성
2. 일반 여성과 아이들
3. 성인 남성

## 호칭 단순화하기

지금은 너무 많은 종류의 호칭이 있어서 패턴을 파악하기 어렵습니다. 비슷한 의미를 가진 호칭들을 그룹화하여 단순화해보겠습니다.

📝 **프롬프트**
```
호칭들을 다음과 같이 단순화해서 새로운 컬럼 Title_Simple을 만들어줘:
- 'Mr'는 그대로
- 'Miss', 'Mlle', 'Ms'는 'Miss'로
- 'Mrs', 'Mme'는 'Mrs'로
- 'Master'는 그대로
- 나머지는 모두 'Rare'로
```

💻 **코드 & 실행결과**
```{code-cell}
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs'
}

for df in [train, test]:
    df['Title_Simple'] = df['Title'].map(title_mapping)
    df['Title_Simple'] = df['Title_Simple'].fillna('Rare')

print("Train data simplified titles:")
print(train['Title_Simple'].value_counts())
print("\nSurvival rate by simplified title:")
print(train.groupby('Title_Simple')['Survived'].mean().sort_values(ascending=False))
```

단순화된 호칭별 생존율을 보면 더 명확한 패턴이 드러납니다:
- 'Rare'(귀족, 성직자 등): 97% 생존
- 'Mrs'(기혼 여성): 79% 생존
- 'Miss'(미혼 여성): 70% 생존
- 'Master'(남자 아이): 57% 생존
- 'Mr'(성인 남성): 16% 생존

이러한 결과는 앞서 발견했던 성별에 따른 생존율 차이를 더욱 세밀하게 보여줍니다. 특히 같은 남성이라도 나이에 따라(Master vs Mr), 같은 여성이라도 결혼 여부에 따라(Mrs vs Miss) 생존율에 차이가 있었다는 점이 흥미롭습니다.

## 모델에 새로운 특성 추가

이제 단순화된 호칭 정보를 원-핫 인코딩으로 변환하여 모델에 추가해보겠습니다.

📝 **프롬프트**
```
Title_Simple을 원-핫 인코딩으로 변환하고 모델을 학습시켜줘
```

💻 **코드 & 실행결과**
```{code-cell}
# Title_Simple 원-핫 인코딩
title_dummies_train = pd.get_dummies(train['Title_Simple'], prefix='Title')
train = pd.concat([train, title_dummies_train], axis=1)

title_dummies_test = pd.get_dummies(test['Title_Simple'], prefix='Title')
test = pd.concat([test, title_dummies_test], axis=1)

# 모델 학습에 사용할 피처 목록 업데이트
inc_fts = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Gender', 
           'Embarked_C', 'Embarked_Q', 'Embarked_S',
           'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare']

# 모델 학습 및 예측
train_and_predict(train, test)
```

호칭 정보를 추가한 모델의 성능은:
- Validation Score: 0.82123 (이전과 동일)
- Public Score: 0.77033 (이전 0.74880에서 크게 상승)
- 418명의 test 데이터 중 약 322명의 생존 여부를 맞혔습니다 (이전보다 9명 증가)

지금까지의 결과를 표로 정리하면:

| 버전 | 피쳐 개수 | Val. Score | Public Score | 맞은 사람 수 |          설명           |
| :--: | :-------: | :--------: | :----------: | :----------: | :---------------------: |
| 1.3  |     5     |  0.73184   |   0.63397    |    265명     | 5개의 numeric 피쳐 사용 |
| 2.1  |     6     |  0.82123   |   0.73444    |    307명     |    Gender 피쳐 추가     |
| 2.2  |     9     |  0.82123   |   0.74880    |    313명     |   승선 항구 정보 추가   |
| 3.1  |    14     |  0.82123   |   0.77033    |    322명     |     호칭 정보 추가      |

호칭 정보의 추가로 예측 정확도가 크게 향상되었습니다. 이는 호칭이 단순한 경칭이 아닌, 승객의 사회적 지위와 상황을 나타내는 중요한 정보였음을 보여줍니다. 다음 섹션에서는 승객의 이름에서 또 다른 중요한 정보인 가족 관계를 추출해보도록 하겠습니다. 같은 성을 가진 승객들은 어떤 패턴을 보일까요?

:::{admonition} 데이터 속 숨은 이야기
:class: seealso

**타이타닉호의 'Master': Thomas Pears**

'Master'라는 호칭에 얽힌 흥미로운 이야기가 있습니다. 보통 'Master'는 어린 남자아이를 지칭했지만, 타이타닉호에는 예외적인 케이스가 있었습니다.

Thomas Pears는 29세의 성인 남성이었지만, 승객 명단에 'Master'로 기록되어 있습니다. 그는 영국의 유명한 비누 제조업체 'A & F Pears'의 후계자였는데, 신혼여행 중이었음에도 'Mr' 대신 'Master'로 등록된 것입니다.

이는 당시 영국 상류층에서 가문의 후계자를 'Master'로 부르는 관습이 있었기 때문입니다. 이러한 예외적인 사례는 호칭이 단순한 나이나 결혼 여부를 넘어, 복잡한 사회적 지위와 관습을 반영했음을 보여줍니다.

안타깝게도 Thomas Pears는 침몰 당시 생존하지 못했습니다. 그의 아내 Edith는 구조되었지만, Thomas의 시신은 끝내 발견되지 않았습니다.
:::