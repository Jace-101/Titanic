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



# 1. 타이타닉 생존자 예측: 첫걸음 떼기

## 데이터 파일 이해하기
데이터 분석 프로젝트를 시작할 때, 가장 먼저 마주하게 되는 것이 바로 데이터 파일들입니다. 타이타닉 생존자 예측 프로젝트에서는 세 가지 파일이 필요합니다. 이 파일들은 캐글(Kaggle)의 타이타닉 대회 페이지(https://www.kaggle.com/competitions/titanic/data)에서 다운로드할 수 있습니다.

**train.csv**

train.csv 파일은 모델을 학습시키기 위한 데이터입니다.
이 파일에는 각 승객의 특징(예: 나이, 성별, 좌석 등)과 함께 해당 승객이 생존했는지 여부(Survived 컬럼)가 기록되어 있습니다. 모델은 이 데이터를 활용해 특징과 생존 여부 사이의 관계를 학습하게 됩니다.
쉽게 말해, train.csv는 문제와 정답이 함께 실린 학습용 교재와 같습니다.

**test.csv**

test.csv 파일은 모델의 성능을 평가하기 위한 데이터입니다.
이 파일에는 승객의 특징은 포함되어 있지만, 생존 여부(Survived)는 제공되지 않습니다. 우리가 만든 모델은 이 데이터를 입력받아 승객들이 생존했을지 여부를 예측해야 합니다.
다르게 표현하자면, test.csv는 배운 내용을 확인하기 위한 시험 문제지와 같습니다.

**gender_submission.csv**

gender_submission.csv 파일은 최종 결과물을 제출하기 위한 파일입니다.
test.csv에서 예측한 생존 여부를 이 파일의 규격에 맞춰 작성한 후, 대회 플랫폼에 제출합니다. 파일에는 승객 ID와 예측한 생존 여부가 포함됩니다.
마치 시험 문제를 풀고 답을 정리해서 제출하는 답안지와 같은 역할을 합니다.

세 파일 간의 관계를 교재-시험-답안지에 비유하면 아래와 같습니다:
- train.csv(교재)로 모델을 학습시킨다.
- 학습된 모델로 test.csv(시험문제) 데이터를 예측한다.
- 예측 결과를 gender_submission.csv(답안지) 형태로 저장해 제출한다.



## 데이터 들여다보기

이제 세 개의 데이터 파일을 불러오겠습니다. 다음과 같이 프롬프트를 입력합니다:

📝 **프롬프트**
```
train.csv, test.csv, gender_submission.csv 파일을 불러와서 
각각 train, test, submission이라는 이름의 데이터프레임으로 저장해줘
```

💻 **코드 & 실행결과**
```{code-cell}
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('gender_submission.csv')
```

위 코드는 세 개의 CSV 파일을 읽어오는 작업을 수행합니다. pd.read_csv()는 CSV(Comma-Separated Values) 파일을 읽어서 데이터프레임으로 변환하는 함수입니다. 데이터프레임은 엑셀의 스프레드시트처럼 행과 열로 구성된 2차원 형태의 데이터 구조입니다. 각각의 파일을 train, test, submission이라는 이름의 데이터프레임 변수에 저장했습니다. 

코드에서 `import pandas as pd`라는 부분이 있는데, 이는 pandas라는 데이터 분석 라이브러리를 불러오면서 'pd'라는 별명을 붙인 것입니다. 그래서 `pd.read_csv()`와 같이 'pd.'을 앞에 붙여서 pandas의 기능을 사용할 수 있는 것입니다.




그런데 위 코드를 실행하면 아무것도 출력되지 않습니다. 코드가 정상적으로 실행되어 데이터를 잘 읽어 들였는지 알 수가 없네요. 확인을 위해 각 데이터의 크기, 즉 행과 열의 수를 출력해보도록 하겠습니다.

📝 **프롬프트**

```
세 데이터프레임의 행과 열 수를 출력해줘
```

💻 **코드 & 실행결과**

```{code-cell}
print("train data shape:", train.shape)
print("test data shape:", test.shape)
print("submission data shape:", submission.shape)
```

위 결과는 다음과 같은 의미를 가집니다:

- train 데이터는 891명의 승객 정보가 있으며, 각 승객마다 12개의 특성이 기록되어 있습니다.
- test 데이터는 418명의 승객 정보가 있고, 11개의 특성이 있습니다. train 데이터보다 특성이 하나 적은 이유는 생존 여부(Survived)가 제외되어 있기 때문입니다.
- submission 데이터는 test 데이터와 같은 418명의 승객에 대해 2개의 열(PassengerId와 Survived)만 가지고 있습니다.




이제 train 데이터의 내용을 자세히 살펴보겠습니다. 다음과 같이 프롬프트를 입력합니다:

📝 **프롬프트**
```
train 데이터의 첫 5개 행을 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
train.head()
```

train 데이터를 살펴보면 각 승객에 대한 다양한 정보가 기록되어 있습니다:
- PassengerId: 승객 번호
- Survived: 생존 여부 (1: 생존, 0: 사망)
- Pclass: 티켓 등급 (1: 1등석, 2: 2등석, 3: 3등석)
- Name: 승객 이름
- Sex: 성별
- Age: 나이
- SibSp: 함께 탑승한 형제자매, 배우자 수
- Parch: 함께 탑승한 부모, 자녀 수
- Ticket: 티켓 번호
- Fare: 요금
- Cabin: 객실 번호
- Embarked: 승선 항구 (C: Cherbourg, Q: Queenstown, S: Southampton)

이렇게 데이터를 살펴보면 우리가 예측에 사용할 수 있는 다양한 정보들이 있다는 것을 알 수 있습니다. 예를 들어, 티켓 등급(Pclass)이나 성별(Sex), 나이(Age) 등은 생존 여부와 관련이 있을 것으로 추측할 수 있습니다.



test 데이터도 살펴보겠습니다. 앞서 train 데이터를 볼 때 사용했던 head() 명령을 활용하면 되겠죠? train을 test로만 바꿔주면 됩니다.

💻 **코드 & 실행결과**
```{code-cell}
test.head()
```

test 데이터를 보면 train 데이터와 매우 비슷하지만, 한 가지 중요한 차이가 있습니다. 바로 'Survived' 열이 없다는 것입니다. 이는 당연한 것인데, test 데이터는 우리가 생존 여부를 예측해야 할 데이터이기 때문입니다.

앞서 교재-시험-답안지 비유를 떠올려보면, test 데이터는 시험 문제지와 같습니다. 시험 문제지에는 답이 적혀있지 않은 것처럼, test 데이터에도 생존 여부가 없는 것입니다. 우리가 만들 인공지능 모델이 바로 이 test 데이터의 승객들에 대해 생존 여부를 예측하게 될 것입니다.



마지막으로 제출 양식인 gender_submission.csv를 살펴보겠습니다. 이제는 익숙하시죠? head() 명령을 사용해서 앞부분 몇 행만 확인해보겠습니다.

💻 **코드 & 실행결과**
```{code-cell}
submission.head()
```

이 파일은 우리가 제출할 답안의 형식을 보여줍니다. PassengerId는 test 데이터와 동일한 승객 번호이고, Survived는 우리가 예측한 생존 여부를 적는 곳입니다.

재미있는 점은 이 파일의 이름이 'gender_submission.csv'라는 것입니다. 보통 다른 캐글 대회의 제출 양식에는 예측해야 할 값이 모두 0으로 되어 있는데, 타이타닉 대회에서는 성별에 기반한 예측 결과를 미리 제공합니다. 이는 타이타닉 대회가 많은 사람들의 '첫 번째 캐글 대회'이기 때문입니다. 초보자들이 참고할 수 있도록 간단한 예시 답안을 제공한 것이죠.



## 결과 제출해보기

우리는 아직 'Survived'를 예측하는 모델을 만들지 않았기 때문에, 이번 섹션에서는 일단 gender_submission.csv 파일을 아무런 변경없이 그대로 제출해 보겠습니다. 캐글의 타이타닉 페이지에서 'Submit Prediction' 버튼을 누르면 아래와 같은 창에서 파일을 올릴 수 있습니다. 'gender_submission.csv' 파일을 올려봅시다.

```{figure} ./images/11-1.png
---
width: 600px
---
결과파일 제출
```


제출결과로 0.76555를 받았습니다. 원고를 쓰는 시점에서 0.76555는 13,315 팀 중 9,643 등이네요. 순위는 'Leaderboard'에서 확인할 수 있습니다.

```{figure} ./images/11-2.png
---
width: 600px
---
제출 결과
```

이 대회에서 사용하는 평가지표는 accuracy 입니다. Accuracy는 분류 문제에서 종종 사용되는 평가지표로 전체 케이스 중에서 정확한 예측을 한 케이스의 비율입니다. 0.76555 * 418 명을 하면 320명이 나오네요. 생존 여부를 정확하게 예측한 경우가 320명이라는 의미입니다. 남자는 0(희생자), 여자는 1(생존자)로 예측한 간단한 모델이지만 꽤 높은 점수가 나왔습니다. 앞으로 계속 학습해 가며, 이 값을 넘는 좋은 예측 모델을 만들어 봅시다!



