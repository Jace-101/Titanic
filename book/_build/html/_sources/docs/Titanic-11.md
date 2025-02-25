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



# 1. 타이타닉 데이터 항해 시작하기

타이타닉 생존자 예측이라는 흥미진진한 여정을 시작합니다. 이 섹션에서는 데이터 분석의 기초를 다지면서, 앞으로 사용하게 될 데이터를 탐색해보겠습니다. 마치 항해를 시작하기 전에 배와 항해 도구들을 점검하는 것처럼, 우리도 데이터와 분석 도구들을 하나씩 살펴볼 것입니다.

이 섹션을 통해 여러분은:
- 데이터 파일들의 역할과 구조를 이해하고
- 판다스(Pandas)를 사용하여 데이터를 읽고 살펴보는 방법을 배우며
- 캐글에 첫 예측 결과를 제출하는 방법을 익히게 됩니다


## 데이터 파일 이해하기 

데이터 분석 프로젝트에서 가장 먼저 해야 할 일은 데이터를 이해하는 것입니다. 타이타닉 생존자 예측 프로젝트는 세 개의 CSV(Comma-Separated Values) 파일로 구성되어 있습니다. 각 파일은 마치 교재, 시험문제, 답안지와 같은 역할을 합니다.

**train.csv: 학습용 데이터**

- 실제 승객들의 특성과 생존 여부가 기록된 데이터입니다
- 모델이 이 데이터를 통해 "어떤 특성을 가진 승객이 생존했는가?"를 학습합니다
- 마치 문제와 답이 함께 있는 교재와 같은 역할을 합니다

**test.csv: 평가용 데이터**

- 승객들의 특성만 제공되고, 생존 여부는 제공되지 않습니다
- 우리가 만든 모델이 이 승객들의 생존 여부를 예측해야 합니다
- 배운 내용을 확인하는 시험문제와 같습니다

**gender_submission.csv: 제출 양식**

- 예측 결과를 제출할 때 사용하는 형식입니다
- 승객 ID와 예측한 생존 여부를 포함합니다
- 시험 답안을 정리해서 제출하는 답안지와 같은 역할입니다

```{figure} ../images/11-data_structure.svg
---
width: 600px
name: data_structure
---
타이타닉 데이터 파일의 구조
```


이제 이 파일들을 실제로 살펴보면서, 데이터 분석의 첫 걸음을 내딛어 보겠습니다.



## 데이터 살펴보기

데이터 분석의 첫 단계는 데이터를 컴퓨터로 읽어들이는 것입니다. 이를 위해 우리는 판다스(Pandas)라는 파이썬 라이브러리를 사용할 것입니다. 판다스는 데이터 분석을 위한 강력한 도구로, 엑셀과 비슷한 형태로 데이터를 다룰 수 있게 해줍니다.

::::{note} 

여러분은 이 책의 모든 코드를 직접 작성할 필요가 없습니다. 각 단계에서 제시되는 프롬프트(📝로 표시)를 입력하면, 필요한 코드가 자동으로 생성됩니다. 이 코드를 복사하여 실행하면 됩니다.

프롬프트는 다음과 같은 도구들에 입력할 수 있습니다:

- **ChatGPT**: chatgpt.com에 접속하여 대화창에 프롬프트를 입력합니다
- **구글 코랩**: 코드 셀에 표시되는 "코딩을 시작하거나 AI로 코드를 <u>생성</u>하세요" 메시지에서 ''<u>생성</u>''을 클릭하면 나타나는 창에 프롬프트를 입력합니다

이는 마치 요리 레시피를 따라하는 것과 같습니다. 요리를 처음 배우는 사람도 레시피만 잘 따라하면 맛있는 요리를 만들 수 있듯이, 코딩을 처음 접하는 분들도 프롬프트만 잘 따라하면 데이터 분석을 수행할 수 있습니다.

각 단계는 다음과 같은 구조로 이루어져 있습니다:

1. 📝 **프롬프트**: 여러분이 입력할 내용
2. 💻 **코드**: 생성된 코드
3. **설명**: 코드가 하는 일과 결과에 대한 설명

::::



<br>

자, 이제 첫 번째 프롬프트를 입력해볼까요?

📝 **프롬프트**
```
우리가 가진 세 개의 파일(train.csv, test.csv, gender_submission.csv)을 읽고,
각각 train, test, submission이라는 이름으로 저장해줘
```

💻 **코드 & 실행결과**
```{code-cell}
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('gender_submission.csv')
```

이 코드는 세 가지 중요한 작업을 수행합니다:
1. `import pandas as pd`: 판다스 라이브러리를 불러옵니다. `pd`는 판다스의 별칭입니다.
2. `pd.read_csv()`: CSV 파일을 읽어오는 판다스의 함수입니다.
3. 읽어온 데이터는 '데이터프레임(DataFrame)'이라는 형태로 저장됩니다. 데이터프레임은 행과 열로 구성된 2차원 테이블로, 엑셀 시트와 비슷합니다.

<br>

데이터를 제대로 불러왔는지 확인하기 위해, 각 데이터프레임의 크기를 살펴보겠습니다.

📝 **프롬프트**
```
방금 만든 train, test, submission 데이터의 크기를 알려줘
```

💻 **코드 & 실행결과**
```{code-cell}
print("train data shape:", train.shape)
print("test data shape:", test.shape)
print("submission data shape:", submission.shape)
```

`shape`는 데이터프레임의 크기를 (행의 수, 열의 수) 형태로 보여주는 속성입니다. 실행 결과를 보면:

- train 데이터: 891명의 승객 정보, 12개의 특성
- test 데이터: 418명의 승객 정보, 11개의 특성 (생존 여부 컬럼 제외)
- submission 데이터: 418명의 승객 정보, 2개의 특성 (ID와 예측할 생존 여부)

<br>

이제 train 데이터의 내용을 살펴보겠습니다. 판다스의 `head()` 메서드를 사용하면 데이터의 첫 5행을 확인할 수 있습니다.

📝 **프롬프트**
```
train 데이터의 내용을 위에서부터 5명만 보여줘
```

💻 **코드 & 실행결과**
```{code-cell}
train.head()
```

각 컬럼은 승객에 대한 서로 다른 특성을 나타냅니다:
- PassengerId: 승객 고유 번호
- Survived: 생존 여부 (1: 생존, 0: 사망)
- Pclass: 객실 등급 (1: 1등석, 2: 2등석, 3: 3등석)
- Name: 승객 이름
- Sex: 성별
- Age: 나이
- SibSp: 함께 승선한 형제자매, 배우자 수
- Parch: 함께 승선한 부모, 자녀 수
- Ticket: 티켓 번호
- Fare: 요금
- Cabin: 객실 번호
- Embarked: 승선 항구 (C: Cherbourg, Q: Queenstown, S: Southampton)



<br>

test 데이터도 살펴보겠습니다. 앞서 train 데이터를 볼 때 사용했던 head() 명령을 활용하면 되겠죠? train을 test로만 바꿔주면 됩니다.

💻 **코드 & 실행결과**
```{code-cell}
test.head()
```

test 데이터를 보면 train 데이터와 매우 비슷하지만, 한 가지 중요한 차이가 있습니다. 바로 'Survived' 컬럼이 없다는 것입니다. 이는 당연한 것인데, test 데이터는 우리가 생존 여부를 예측해야 할 데이터이기 때문입니다.

앞서 교재-시험-답안지 비유를 떠올려보면, test 데이터는 시험 문제지와 같습니다. 시험 문제지에는 답이 적혀있지 않은 것처럼, test 데이터에도 생존 여부가 없는 것입니다. 우리가 만들 인공지능 모델이 바로 이 test 데이터의 승객들에 대해 생존 여부를 예측하게 될 것입니다.

<br>

마지막으로 제출 양식인 gender_submission.csv를 살펴보겠습니다. 이제는 익숙하시죠? head() 명령을 사용해서 첫 5행만 확인해보겠습니다.

💻 **코드 & 실행결과**
```{code-cell}
submission.head()
```

이 파일은 우리가 제출할 답안의 형식을 보여줍니다. PassengerId는 test 데이터와 동일한 승객 번호이고, Survived는 우리가 예측한 생존 여부를 적는 곳입니다.

재미있는 점은 이 파일의 이름이 'gender_submission.csv'라는 것입니다. 보통 다른 캐글 대회의 제출 양식에는 예측해야 할 값이 모두 0으로 되어 있는데, 타이타닉 대회에서는 성별에 기반한 예측 결과를 미리 제공합니다. 이는 타이타닉 대회가 많은 사람들의 '첫 번째 캐글 대회'이기 때문입니다. 초보자들이 참고할 수 있도록 간단한 예시 답안을 제공한 것이죠.



## 결과 제출하기

우리는 아직 'Survived'를 예측하는 모델을 만들지 않았지만, 타이타닉 대회에 첫 제출을 해보도록 하겠습니다. 이번 섹션에서는 일단 gender_submission.csv 파일을 아무런 변경없이 그대로 제출해 보겠습니다. 이 파일은 성별에 기반한 단순한 예측을 담고 있습니다. 즉, 여성 승객은 생존(1), 남성 승객은 사망(0)으로 예측한 결과입니다.

[캐글의 타이타닉 페이지](https://www.kaggle.com/competitions/titanic)에서 'Submit Prediction' 버튼을 누르면 파일을 제출할 수 있습니다. gender_submission.csv 파일을 제출하면 0.76555라는 점수를 받게 됩니다. 이는 418명의 테스트 데이터 중 약 320명(0.76555 × 418 ≈ 320)의 생존 여부를 정확하게 예측했다는 의미입니다.

```{figure} ../images/11-1.png
---
width: 600px
name: submit-prediction
---
예측결과 제출
```

```{figure} ../images/11-2.png
---
width: 600px
name: leaderboard
---
제출결과 확인
```

이렇게 단순히 성별만으로도 76.555%의 정확도를 달성할 수 있다는 점이 흥미롭습니다. 이는 타이타닉 침몰 당시 "여성과 아이 먼저"라는 원칙이 실제로 지켜졌음을 보여주는 증거이기도 합니다.

이제 기본적인 데이터 구조를 이해했으니, 다음 섹션에서는 본격적으로 데이터를 분석하면서 더 나은 예측 모델을 만들어보도록 하겠습니다.



:::{note}
캐글의 타이타닉 대회에서 사용하는 평가 지표는 accuracy(정확도)입니다. accuracy는 전체 예측 중 올바른 예측의 비율을 의미합니다. 이후 챕터에서 accuracy를 포함한 다양한 평가 지표들에 대해 자세히 다루게 될 것입니다.
:::



:::{admonition} 직접 해보기

train 데이터의 내용을 살펴보는 간단한 방법들을 실습해보세요:

- `train.head(3)`: 위에서부터 3명의 정보를 보여줍니다. head() 안에 숫자를 넣어 원하는 만큼의 행을 볼 수 있습니다.
- `train.tail()`: 마지막 5행을 보여줍니다
- `train`: 데이터프레임 이름만 입력하면 전체 데이터를 보여줍니다 (단, 행이 많을 경우 앞뒤 일부만 표시)

:::



:::{admonition} 데이터 속 숨은 이야기 
:class: seealso

**데이터 사이언스의 "Hello World"**

캐글에서 가장 유명한 입문용 대회가 바로 이 타이타닉 생존자 예측 대회입니다. 현재까지도 전 세계의 많은 데이터 사이언티스트들이 이 대회로 머신러닝을 시작하고 있죠.

이 대회에서 사용되는 데이터는 실제 타이타닉호의 승객 정보를 바탕으로 교육 목적에 맞게 가공된 것입니다. Age나 Cabin 같은 중요한 정보들의 결측치를 그대로 남겨둔 것도 이러한 교육적 목적 때문이었습니다. 현실 세계의 데이터는 완벽하지 않다는 것을 보여주기 위해서죠.

실제로 많은 데이터 사이언티스트들이 이 대회를 통해 결측치 처리, 범주형 변수 인코딩 등 데이터 전처리의 기초를 배웠다고 합니다. 프로그래밍 입문자들이 "Hello World"로 시작하듯, 데이터 사이언스 입문자들은 타이타닉 대회로 시작합니다. 그래서 이 대회를 "데이터 사이언스의 Hello World"라고 부르기도 합니다.

:::

<br>
