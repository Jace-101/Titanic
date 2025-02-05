# Jupyter Book으로 책 만들기

## Build

- Anaconda Prompt에서 해당 폴더로 이동 (C:\Users\mlee\quantecon-mini-example\Titanic)

- \Titanic\book 폴더에 원고 파일(*.md) 생성

- \Titanic\book 폴더에 세 개 데이터 파일 복사

- \Titanic\book\images 폴더에 화면 캡쳐한 파일 복사 

- 다음 명령을 실행해 빌드

  ```
  jupyter-book build book
  ```

- 목차 파일 (_toc.yml) 수정

- (필요 시) (_config.yml) 수정



## Hosting

- cd book

- 아래 명령어 실행

  ```
  git add .
  git commit -m "adding my first book!"
  git push
  ```

- cd ..

  ```
  ghp-import -n -p -f book/_build/html -m "initial publishing"
  ```

- 결과 확인: https://jace-101.github.io/Titanic/intro.html