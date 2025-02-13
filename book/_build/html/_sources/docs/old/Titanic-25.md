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



# 5. 나이(Age)는 상관계수가 낮던데, 생존자 예측에 얼마나 도움이 될까?





2.5 섹션의 원고를 작성해드리겠습니다. 먼저 첫 번째 코드셀에 대한 내용을 작성하겠습니다. Age 변수의 결측치를 처리하고 승객의 생존율을 분석하는 내용이네요.

## Age 결측치 현황 파악

나이는 승객의 가장 기본적인 특성 중 하나지만, 생존율과의 상관계수는 -0.07로 매우 낮게 나타났습니다. 하지만 이는 결측치가 있는 상태에서 계산된 값이므로, 먼저 결측치 현황을 정확히 파악할 필요가 있습니다.

📝 **프롬프트**
```
train과 test 데이터의 Age 컬럼 결측치 개수를 계산해줘
```

💻 **코드 & 실행결과**
```{code-cell}
print(f"Number of missing values in Age (train): {train['Age'].isnull().sum()}")
print(f"Number of missing values in Age (test): {test['Age'].isnull().sum()}")
```

이 코드는 Age 컬럼의 결측치 개수를 확인합니다. train 데이터와 test 데이터 각각에서 isnull() 함수로 결측치 여부를 확인하고, sum() 함수로 결측치의 총 개수를 계산합니다.

실행 결과를 보면 train 데이터에는 177개, test 데이터에는 86개의 결측치가 있음을 알 수 있습니다. 이는 train 데이터의 약 20%, test 데이터의 약 21%에 해당하는 상당히 많은 양입니다. 이렇게 많은 결측치가 있다면 단순히 제거하기보다는 적절한 값으로 채우는 것이 좋겠죠.



## 연령대별 데이터 분석 시작

먼저 Age 데이터를 10살 단위로 그룹화하고, 각 그룹의 승객 수를 확인해보겠습니다.

📝 **프롬프트**
```
train 데이터의 Age를 10살 단위로 그룹화하고, 각 연령대별 승객 수를 출력해줘
```

💻 **코드 & 실행결과**
```{code-cell}
import pandas as pd

# Age를 10살 단위로 그룹화
train['AgeGroup'] = train['Age'] // 10 * 10
age_stats = train.groupby('AgeGroup')['PassengerId'].count().reset_index()
print(age_stats)
```

실행 결과를 보면 60대 이상 승객의 수가 매우 적음을 알 수 있습니다. 이런 경우 데이터가 너무 적은 그룹들은 하나로 합치는 것이 좋습니다. 60세 이상을 하나의 그룹으로 통합하겠습니다.

📝 **프롬프트**
```
60세 이상을 하나의 그룹으로 묶은 다음, 연령대별 승객 수를 막대그래프로 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
import plotly.express as px

# 60세 이상을 하나의 그룹으로 통합
train.loc[train['Age'] >= 60, 'AgeGroup'] = 60
age_stats = train.groupby('AgeGroup')['PassengerId'].count().reset_index()

# 막대그래프 생성
fig = px.bar(age_stats, 
             x='AgeGroup', 
             y='PassengerId',
             title='Number of Passengers by Age Group',
             labels={'PassengerId': 'Number of Passengers', 
                    'AgeGroup': 'Age Group'})

# x축 레이블 수정
fig.update_xaxes(ticktext=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+'],
                 tickvals=[0, 10, 20, 30, 40, 50, 60])

fig.show()
```

막대그래프를 통해 승객 분포를 한눈에 파악할 수 있습니다:
- 20-29세 승객이 가장 많으며, 전체의 약 30%를 차지합니다
- 0-9세 어린이는 전체의 약 8%입니다
- 60세 이상의 고령 승객은 전체의 약 5%로 가장 적습니다

📝 **프롬프트**
```
각 연령대별 생존율을 꺾은선 그래프로 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
# 그룹별 생존율 계산
age_survival = train.groupby('AgeGroup')['Survived'].mean().reset_index()

# 꺾은선 그래프 생성
fig = px.line(age_survival, 
              x='AgeGroup', 
              y='Survived',
              title='Survival Rate by Age Group',
              labels={'Survived': 'Survival Rate', 
                     'AgeGroup': 'Age Group'},
              markers=True)  # 데이터 포인트 표시

# x축 레이블 수정
fig.update_xaxes(ticktext=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+'],
                 tickvals=[0, 10, 20, 30, 40, 50, 60])
fig.update_yaxes(range=[0, 1])

fig.show()
```

꺾은선 그래프를 통해 연령대별 생존율의 변화 추이를 더 명확하게 볼 수 있습니다:
- 0-9세 어린이의 생존율이 약 60%로 가장 높습니다
- 20대의 생존률이 약 35%로, 60대 이상을 제외하면 가장 낮은 수준을 보입니다
- 30대와 40대에서는 생존률이 다시 상승하고, 50대에서 하락하는 등 연령대별로 등락을 반복합니다
- 60세 이상 고령층의 생존율이 약 30%로 가장 낮습니다

이러한 분석 결과는 앞서 살펴본 Age와 생존율 사이의 낮은 상관계수(-0.07)를 설명해줍니다. 나이가 증가할수록 생존율이 단순히 증가하거나 감소하는 것이 아니라, 연령대별로 복잡한 패턴을 보이기 때문입니다. 특히 어린이의 높은 생존율과 20대의 낮은 생존율이 대조를 이루고, 중년층에서 다시 생존율이 상승하는 등의 비선형적 관계가 관찰됩니다. 이는 Age 변수가 생존 예측에 있어 상관계수가 시사하는 것보다 더 중요한 역할을 할 수 있음을 의미합니다. 따라서 결측치를 채울 때도 이러한 연령대별 특성을 고려한 더 정교한 접근이 필요할 것으로 보입니다.



## Age와 다른 특성들의 관계 분석

앞서 우리는 나이와 생존율의 관계가 단순한 선형관계가 아님을 확인했습니다. 이제 Age 변수의 결측치를 채우기 위해, Age와 다른 특성들 간의 관계를 살펴보겠습니다.

📝 **프롬프트**
```
Age와 다른 수치형 변수들(Pclass, SibSp, Parch, Fare, Gender) 사이의 상관계수를 계산해줘
```

💻 **코드 & 실행결과**
```{code-cell}
numeric_cols = ['Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Gender']
corr = train[numeric_cols].corr()['Age'].sort_values()
print(corr)
```

상관계수를 통해 몇 가지 흥미로운 패턴이 발견됩니다:
- Pclass와 가장 높은 상관관계(-0.37)를 보입니다
- 부호가 음수인 것은 좌석 등급이 높을수록(Pclass 값이 작을수록) 나이가 많은 경향이 있다는 의미입니다
- 다른 변수들과는 상대적으로 약한 상관관계를 보입니다

이 결과는 Age의 결측치를 채울 때 Pclass를 활용하는 것이 가장 효과적일 수 있음을 시사합니다. 즉, 나이가 누락된 승객의 경우, 같은 좌석 등급의 다른 승객들의 나이를 참고하면 좋을 것 같습니다. 

다음 섹션에서는 각 좌석 등급별 승객들의 중앙값 나이로 결측치를 채우는 방법을 시도해보겠습니다. 중앙값을 사용하는 이유는 평균값보다 극단값의 영향을 덜 받기 때문입니다. 이렇게 하면 좌석 등급별 특성을 반영하면서도, 안정적인 결측치 처리가 가능할 것으로 기대됩니다.



## Pclass별 Age 분포 확인

결측치를 채우기 전에, 각 좌석 등급별로 승객들의 나이 분포를 살펴보겠습니다.

📝 **프롬프트**
```
각 좌석 등급(Pclass)별 Age의 중앙값을 계산해줘
```

💻 **코드 & 실행결과**
```{code-cell}
age_by_pclass = train.groupby('Pclass')['Age'].median()
print("Median Age by Pclass:")
print(age_by_pclass)
```

각 좌석 등급별 나이의 중앙값을 보면:
- 1등석 승객: 약 37세
- 2등석 승객: 약 29세
- 3등석 승객: 약 24세

예상대로 좌석 등급이 높을수록 승객의 연령대도 높아지는 것을 확인할 수 있습니다. 이제 이 중앙값들을 사용하여 결측치를 채워보겠습니다.

📝 **프롬프트**
```
각 Pclass별 중앙값으로 Age의 결측치를 채우고, 결과를 확인해줘
```

💻 **코드 & 실행결과**
```{code-cell}
# train 데이터의 결측치 채우기
for pclass in [1, 2, 3]:
    train.loc[(train['Age'].isnull()) & (train['Pclass'] == pclass), 'Age'] = age_by_pclass[pclass]

# test 데이터의 결측치 채우기
for pclass in [1, 2, 3]:
    test.loc[(test['Age'].isnull()) & (test['Pclass'] == pclass), 'Age'] = age_by_pclass[pclass]

# 결과 확인
print("Train data missing values:", train['Age'].isnull().sum())
print("Test data missing values:", test['Age'].isnull().sum())
```

이 코드는 다음과 같은 작업을 수행합니다:
1. train과 test 데이터 각각에 대해
2. 각 좌석 등급별로 나이가 누락된 승객들을
3. 해당 등급의 중앙값 나이로 채웁니다

실행 결과를 보면 모든 결측치가 성공적으로 채워졌음을 알 수 있습니다. 이렇게 좌석 등급별 특성을 반영하여 결측치를 채움으로써, 단순히 전체 평균이나 중앙값으로 채우는 것보다 더 현실적인 값을 얻을 수 있었습니다.

이제 남은 과제는 이렇게 처리된 Age 변수가 실제로 생존 예측의 정확도를 얼마나 향상시킬 수 있는지 확인하는 것입니다. 다음 섹션에서는 지금까지 처리한 모든 변수들을 사용하여 새로운 예측 모델을 만들어보도록 하겠습니다.



마지막 단락을 아래와 같이 수정하고 결과에 대한 설명을 추가하겠습니다.

이렇게 처리된 데이터로 Random Forest 모델을 학습시킨 결과:
- Validation Score는 0.82123으로 이전과 동일했습니다
- Public Score는 0.75598로, 이전 버전(0.74880)보다 소폭 상승했습니다
- 418명의 test 데이터 중 약 316명의 생존 여부를 정확하게 예측했습니다

이 결과는 매우 흥미로운 시사점을 제공합니다. 먼저 Validation Score가 이전과 동일하다는 것은 Age 결측치 처리 방식이 train 데이터 내에서의 예측 성능에는 큰 영향을 미치지 않았다는 것을 의미합니다. 하지만 Public Score가 상승한 것은 이 방식이 실제 test 데이터에 대해서는 더 나은 예측을 가능하게 했다는 것을 보여줍니다.

지금까지의 결과를 표로 정리하면 다음과 같습니다:

| 버전 | 피쳐 개수 | Val. Score | Public Score | 맞은 사람 수 |               설명                |
| :--: | :-------: | :--------: | :----------: | :----------: | :-------------------------------: |
| 1.3  |    5개    |  0.73184   |   0.63397    |    265명     |      5개의 numeric 피쳐 사용      |
| 2.1  |    6개    |  0.82123   |   0.73444    |    307명     |         Gender 피쳐 추가          |
| 2.2  |    9개    |  0.82123   |   0.74880    |    313명     |        탑승항구 정보 추가         |
| 2.5  |    9개    |  0.82123   |   0.75598    |    316명     | Age 결측치를 Pclass별로 대체 처리 |

점진적인 성능 향상을 통해, 우리의 데이터 전처리 방식이 올바른 방향으로 진행되고 있음을 확인할 수 있습니다.



:::{admonition} 그들의 나이
:class: note

영화 '타이타닉'의 실제 모델이 된 승객들의 나이를 살펴보면:

- **Molly Brown(생존)**: 44세
  - "침몰하지 않는 Molly"로 알려진 그녀는 1등실 승객이었습니다
  - 자신이 탄 구명보트를 다시 침몰 지점으로 돌리자고 주장했으나 받아들여지지 않았습니다
  - 영화에서는 Kathy Bates(당시 48세)가 연기했습니다

- **Thomas Andrews(사망)**: 39세
  - 타이타닉호의 설계자였던 그는 마지막까지 승객들의 탈출을 도왔습니다
  - 영화에서는 Victor Garber(당시 47세)가 연기했습니다
  - 우리의 데이터에서 그의 요금은 0으로 기록되어 있습니다

반면 영화 속 허구의 인물들:
- Rose DeWitt Bukater: 17세 (Kate Winslet은 당시 21세)
- Jack Dawson: 20세 (Leonardo DiCaprio는 당시 22세)

실제 역사적 인물들의 나이는 우리가 앞서 분석한 연령대별 생존율과 흥미로운 연관성을 보입니다. 40대의 Molly Brown은 1등실 승객이라는 이점이 있었고, 반면 30대의 Andrews는 마지막까지 승객들을 돕다가 침몰했습니다. 이는 당시 생존이 단순히 나이에 의해서만 결정된 것이 아니었음을 보여줍니다. 특히 Molly Brown의 사례는 객실 등급(Pclass)이 나이(Age)보다 생존에 더 큰 영향을 미쳤을 수 있다는 우리의 분석 결과와도 일치합니다.
:::







