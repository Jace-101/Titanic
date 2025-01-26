# 1.1 첫 도전: 직관으로 시작하는 생존자 예측

## Setup: 기본 준비하기
데이터 분석을 시작하려면 몇 가지 도구가 필요합니다. 아래 코드는 지금은 이해하기 어려울 수 있습니다. 실행을 위해 필요한 준비 과정이라고 생각하고 넘어가도 좋습니다.

**[코드]**
```python
%%time
# ===== 패키지 설치 =====
!pip install dask[dataframe] -q
```

**[출력]**
```output
CPU times: user 92.5 ms, sys: 11.7 ms, total: 104 ms
Wall time: 10.8 s
```

**[코드]**
```python
%%time
# ===== 데이터 분석을 위한 라이브러리 =====
import numpy as np
import pandas as pd
[... 나머지 import 구문 ...]
```

## Data Overview: 데이터 살펴보기
먼저 캐글 사이트(https://www.kaggle.com/competitions/titanic/data)에서 타이타닉 데이터를 다운로드 받습니다. 그 다음, 다운로드 받은 파일들을 코랩에 업로드해야 합니다. 아래 코드를 실행하면 '파일 선택' 버튼이 나타납니다. 이 버튼을 클릭한 후, train.csv, test.csv, gender_submission.csv 세 개의 파일을 한 번에 선택해서 업로드해주세요.

**[코드]**
```python
from google.colab import files
uploaded = files.upload()
```

**[출력]**
```output
Saving train.csv to train.csv
Saving test.csv to test.csv
Saving gender_submission.csv to gender_submission.csv
```

이제 업로드한 파일들을 불러와보겠습니다.

**[코드]**
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('gender_submission.csv')
```

데이터가 제대로 준비되었는지 확인해보겠습니다.

**[코드]**
```python
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)
print("Submission data shape:", submission.shape)
```

**[출력]**
```output
Train data shape: (891, 12)
Test data shape: (418, 11)
Submission data shape: (418, 2)
```

이제 데이터가 어떻게 구성되어있는지 자세히 살펴보겠습니다.

**[코드]**
```python
train.head()
```

**[출력]**
```output
   PassengerId  Survived  Pclass                                             Name     Sex   Age  SibSp  Parch     Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0  A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0   PC 17599  71.2833   C85        C
2            3         1       3                              Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2...   7.9250   NaN        S
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0    113803   53.1000  C123        S
4            5         0       3                            Allen, Mr. William Henry    male  35.0      0      0     373450   8.0500   NaN        S
```

train 데이터의 각 열(column)은 다음과 같은 정보를 담고 있습니다:
- PassengerId: 승객 번호
- Survived: 생존 여부 (0=사망, 1=생존)
- Pclass: 객실 등급 (1=1등급, 2=2등급, 3=3등급)
- Name: 승객 이름
- Sex: 성별
- Age: 나이
- SibSp: 함께 탑승한 형제자매, 배우자 수
- Parch: 함께 탑승한 부모, 자녀 수
- Ticket: 티켓 번호
- Fare: 티켓 요금
- Cabin: 객실 번호
- Embarked: 탑승 항구 (C=Cherbourg, Q=Queenstown, S=Southampton)

test 데이터도 살펴보겠습니다. 위에서 train.head()를 사용해봤으니, 이번에는 train을 test로 바꾸기만 하면 됩니다.

**[코드]**
```python
test.head()
```

**[출력]**
```output
   PassengerId  Pclass                                          Name     Sex   Age  SibSp  Parch   Ticket     Fare Cabin Embarked
0          892       3                              Kelly, Mr. James    male  34.5      0      0   330911   7.8292   NaN        Q
1          893       3              Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0   363272   7.0000   NaN        S
2          894       2                     Myles, Mr. Thomas Francis    male  62.0      0      0   240276   9.6875   NaN        Q
3          895       3                              Wirz, Mr. Albert    male  27.0      0      0   315154   8.6625   NaN        S
4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist) female  22.0      1      1  3101298  12.2875   NaN        S
```

test 데이터를 보니 train 데이터와 비교해서 'Survived' 피쳐가 없네요. 왜 그럴까요? test 데이터는 우리가 생존 여부를 예측해야 할 승객들의 정보이기 때문입니다. 실제 타이타닉 사고 당시 이 승객들의 생존 여부는 알려져 있지만, 캐글은 이를 공개하지 않고 우리의 예측이 얼마나 정확한지 평가하는 기준으로 사용합니다.

마지막으로 제출 양식을 살펴보겠습니다.

**[코드]**
```python
submission.head()
```

**[출력]**
```output
   PassengerId  Survived
0          892         0
1          893         1
2          894         0
3          895         0
4          896         1
```

## 첫 번째 예측 도전하기
캐글에서 제공하는 gender_submission.csv 파일은 아주 단순한 규칙으로 만들어졌습니다:
- 모든 여성 승객 → 생존(1)으로 예측
- 모든 남성 승객 → 사망(0)으로 예측

이 파일을 제출하면 놀랍게도 76.555%라는 꽤 높은 정확도가 나옵니다. 이는 "여성과 아이를 먼저 구하라"는 당시의 구조 원칙이 실제로 잘 지켜졌다는 것을 보여줍니다.

## 다음 단계는?
76.555%! 첫 도전치고는 훌륭한 성적입니다. 하지만 더 높은 정확도를 얻을 수 있지 않을까요? 다음 시간에는 이 점수를 높이는 방법을 찾아보겠습니다.