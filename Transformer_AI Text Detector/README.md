# RoBERTa 기반 작성글 AI 판별기

사람 답변과 ChatGPT 답변을 학습해서, 입력한 글의 **AI 의심 점수**와 **의심 구역**을 보여주는 프로젝트입니다.

- 모델: [klue/roberta-base](https://huggingface.co/klue/roberta-base)
- 데이터셋: [nayohan/HC3-ko](https://huggingface.co/datasets/nayohan/HC3-ko)
- 노트북: `RoBERTa.ipynb` (Google Colab, GPU)

---

## 1. RoBERTa란

### 모델이 발표된 배경
- BERT의 학습 방식을 개선하기 위해 나온 모델 (Robustly Optimized BERT Pretraining Approach, 2019년 Facebook AI)
- BERT가 충분히 학습되지 않은 모델이었다는 문제의식에서 출발해, 구조는 유지하고 학습 방법을 최적화함
- 더 많은 데이터로 오래 학습하면서 성능을 높였음

### Encoder / Decoder / Encoder-Decoder 중 어떤 구조인지
- Encoder 구조
- 문장의 앞뒤 내용을 같이 보고 의미를 파악함

### 사전학습 방식
- 문장 일부를 가리고 어떤 단어인지 맞히는 MLM 방식
- 가리는 위치는 학습하면서 바뀌고(dynamic masking), BERT에서 사용하던 다음 문장 예측(NSP)은 제외함

### 기존 Transformer와 달라진 점
- 기존 Transformer는 Encoder와 Decoder를 같이 사용하지만, RoBERTa는 Encoder만 사용함
- BERT와 비교하면 NSP 제거, dynamic masking 적용, 더 큰 배치와 더 긴 학습, 학습 데이터 증가(약 16GB → 약 160GB), 토크나이저 변경(WordPiece → byte-level BPE) 등이 있음
  - 위 내용은 원조 RoBERTa 기준입니다. 이 프로젝트의 klue/roberta-base는 `BertTokenizer`(vocab 32,000)를 사용합니다.

### 해당 모델이 잘하는 작업
- 감정 분석, 텍스트 분류, 문장 유사도 분석, 질의응답 등에 사용
- 텍스트 생성보다는 문장을 이해하는 작업에 적합
- GLUE, SQuAD 같은 벤치마크에서 BERT보다 높은 성능을 보임

### 모델의 장점과 단점
- 장점: 같은 구조의 BERT보다 성능이 더 높으며, 사전학습된 모델을 가져와 내 데이터로 추가 학습(파인튜닝)할 수 있음
- 단점: 입력 길이가 제한(최대 512 토큰)되며, 학습한 데이터와 다른 글에서는 잘못 판단할 수 있음. 또한 사전학습 비용이 크고 생성 작업에는 적합하지 않음

---

## 2. 작성글 AI 판별기

### 모델과 데이터셋 소개
- klue/roberta-base와 nayohan/HC3-ko를 사용함
- 사람 답변과 ChatGPT 답변을 학습해서, 입력한 글의 AI 의심 점수와 의심 구역을 보여주는 프로젝트
- HC3-ko는 영어 데이터셋 HC3를 한국어로 번역한 데이터셋 (전체 24,322행)

### Tokenizer를 이용한 데이터 전처리
- 문자열로 저장된 답변 목록을 리스트로 바꾸고, URL 자리표시자와 불필요한 공백을 정리함
- Tokenizer로 모델에 넣을 토큰 ID를 만들고, 최대 512토큰으로 자름

### 사전학습 모델 Fine-tuning
- 질문마다 사람 답변과 ChatGPT 답변을 하나씩 사용함 (8,961쌍, 17,922개 글)
- 사람은 0, ChatGPT는 1로 설정해서 두 종류의 글을 구분하도록 추가 학습함

### 학습 과정 및 성능 평가
- 같은 질문이 학습과 평가에 겹치지 않게 질문 단위로 데이터를 나눔 (Train 7,156쌍 / Val 904쌍 / Test 901쌍)
- 에폭마다 Train과 Val의 Loss, Accuracy, Precision, Recall, F1 score를 확인함
- 그래프를 참고하면 Val Loss는 불안정한 모습을 보임
- Accuracy는 Train/Val 모두 높게 나왔으며 과적합이 의심됨
- Precision, F1은 불안정한 모습을 보였고, Recall은 3에폭에서 살짝 튀는 현상이 있었지만 거의 1에 수렴함
- Val F1이 가장 높은 모델을 저장하고, Test 데이터로 최종 성능을 확인함

<!-- 학습 그래프 이미지를 추가하려면 아래 줄의 주석을 풀고 경로를 맞춰주세요.
![Loss](images/loss.png)
![Accuracy](images/accuracy.png)
![Validation Performance](images/val_performance.png)
-->

**학습 설정**

| 항목 | 값 |
|---|---|
| 최대 입력 길이 | 512 토큰 |
| 배치 크기 | 16 |
| 에폭 | 10 (약 11분, Colab A100) |
| 학습률 / Weight decay | 2e-5 / 0.01 |
| Warmup 비율 | 0.10 |
| Optimizer | AdamW (Mixed Precision 사용) |

**결과**

| 구분 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Val (Best, 8에폭) | 0.9845 | 0.9740 | 0.9956 | 0.9847 |
| Test | 0.9856 | 0.9740 | 0.9978 | 0.9857 |

### 새로운 문장을 직접 입력하여 예측
- `my_text`에 글을 넣고 `show_detection(my_text)`를 실행하면 됨
- 각 문장을 앞뒤 1문장과 묶어 문장별 의심 점수를 계산하고, 평균을 전체 의심 점수로 보여줌
- 기준(0.5) 이상인 문장은 붉은색, 그렇지 않으면 초록색으로 표시함

```python
my_text = """
여기에 판별할 글을 입력
"""

show_detection(my_text)
```

### 모델의 문제점 및 개선 방법 분석
- 직접 ChatGPT에 질문한 후 답변을 받아 검증했을 때,
  - 데이터셋(HC3)의 ChatGPT 답변은 2023년 초에 수집된 것이라, 현재 ChatGPT와는 모델 버전이 달라 차이가 있을 수 있음
  - Gemini, Claude는 데이터셋이 없어 추가하지 못했지만, 데이터셋을 만들거나 나온다면 추가해 학습할 수 있음
  - Gemini, Claude의 답변을 넣은 결과, GPT의 답변을 넣었을 때와 큰 차이를 느끼지 못함
  - 그래서 사람과 AI에 대한 구분을 잘한다고 보기는 어려움
- 작성자가 직접 쓴 글은 모델이 사람이 작성했다고 판별함
- 하지만 글을 더 길게 작성할수록 AI가 작성했다고 판별할 가능성이 커진다고 느낌
- 의심 점수가 높다고 해서 실제 ChatGPT가 작성한 글이라고 확정할 수는 없음
- 저장된 `.pt`를 가져와서 테스트를 진행해도 되지만, 학습 시간이 길지 않고 성능도 충분히 좋다고 하기 어렵기 때문에, 공부할 겸 모델을 직접 분석하고 훈련해 보는 것도 좋을 것 같음

---

## 실행 방법

1. Google Colab에서 `RoBERTa.ipynb`를 열고 GPU 런타임을 선택
2. 필요한 라이브러리 설치

   ```bash
   pip install transformers datasets accelerate scikit-learn
   ```

3. 셀을 위에서부터 순서대로 실행 (학습된 모델은 `HC3_ko_best_model.pt`로 Google Drive에 저장됨)
4. 마지막의 `my_text`에 글을 넣고 `show_detection(my_text)` 실행
