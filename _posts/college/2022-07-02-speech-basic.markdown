---
layout: post
title: "Speech Basic"
image: speech1.jpg #12.jpg
date: 2022-07-02 12:00:18 +0200
tags: [ai, sound-ai]
categories: [sound-ai]
---

----
## 0.0.0 Baisc concept

#### 채널(Channel)
오디오 채널이란 녹음/재생하는 신호의 수를 의미

##### Mono(Monophonic sound): 하나의 마이크 혹은 스피커와 같이 1개의 채널을 통해 듣는 소리.

##### Stero(Sterophonic sound): 스피커의 대칭 구성을 통해, 둘 이상의 독립 음향 채널을 사용하는 음향 재생 방식. streo로 구성된 오디오는 왼쪽,오른쪽 이 두개의 채널의 소리가 미세하게 다르게되어,소리가 공간감 있고 풍부하게 들린다.

##### Speaker Channel: 소리가 나오는 스피커가 몇개인지를 말하며, 채널이 많으면 소리가 여러 스피커를 통해 분산해 나온다고 생각하면 된다.

##### Harmonics(배음)구조
소리는 한 가지의 주파수만으로 구성되지 않기 때문에, 기본 주파수(Fundamental frequency)와 함께 기본 주파수의 정수배인 배음(harmonics)들로 구성된다. 
예를들어 피아노 건반에서 4옥타브 '라'(440Hz)음을 연주했다면 그소리는 기본주파수인 440Hz뿐만 아니라, 그 정수배인 880Hz, 그리고 그 다음 배음들까지 포함한다.

##### Formants(포먼트)
소리가 공명되는 특정 주파수 대역이다.
사람의 음성은 Vocal folds(성대)애서 형성되어, Vocal track(성도)를 거치며 변형되는데, 소리는 성도를 지날때, 포먼트를 만나 증폭되거나 감쇠된다. 즉. Formant(포먼트)는 Harmonics 배음과 만나 소리를 풍성하게 혹은 선명하게 만드는 필터역할을 한다.

##### 정현파 (Sinusoid)
모든 신호는 주파수(frequency)와 크기(magnitude), 위상(phase)이 다른 정현파(sinusolida signal)의 조합으로 나타낼 수 있다. 퓨리에 변환은 조합된 정현파의 합(하모니) 신호에서 그 신호를 구성하는 정현파(주기적인 파형)들을 각각 분리해내는 방법이다.  
오일러 공식: $$ e^jθ=cosθ+jsinθ $$ 

##### cepstrum
사람의 목소리는 기본 주파수(F0 : fundamental frequency)와 배음(harmonics)으로 이루어진다. 기본 주파수는 정현파(사인파) 요소 중 가장 작은 주파수이고, 배음은 기본 주파수의 정수배로 발생한다. 소리가 공명되는 특정 주파수를 음형(formant)라고 하는데, 제1(F1), 제2 음형대(F2)의 주파수에 따라 모음이 달라지고 이러한 현상을 조음(articulation)이라고 한다.  
하지만 공명 주파수가 배음 주파수와 어긋나면 formant를 찾기가 어렵다. 따라서 Cepstrum을 통해 F0을 구하고 이를 통해 F0의 정수배인 harmonic peak를 분리할 수 있다. Log-spectrum에 역푸리에 변환(IFFT)을 취하면, Cepstrum을 구할 수 있다.

## Summary: Spectrogram이란 ?


Spectrogram의 x축은 시간 축(단위: frame), y축은 주파수를 의미한다. 그리고 각 시간당 주파수가 가지는 값을 값의 크기에 따라 색으로 표현하여 3차원을 2차원으로 표현하게 된다. 

Spectrogram을 추출하는 일반적으로 프로세스는 입력신호에 대해서 window function을 통과하여 window size만큼 sampling 된 data를 받아서, Discrete Fourier Transform을 거치게 됩니다. DFT를 거친 신호들은 Frequency와 Amplitude의 영역을 가지는 Spectrum이 됩니다. 이후 이를 90도로 회전시켜서, time domain으로 stack하게 됩니다. 혹은 Windowing 과정에서 다뤘던 데이터에 대해 STFT로 power-spectrum을 구해 여기에 데시벨 변환 공식을 취해서 log-spectrum을 구하고 이를 세로로 세워서 frame마다 차곡차곡 옆으로 쌓으면, 푸리에 변환으로 사라졌던 time domain을 복원할 수 있고 이를 Spectrogram이라고 한다.

## Amplitude envelope
- Max amplitude value of all samples in a frame
- Gives rough idea of loudness
- Sensitive to outliers
- Onset detection, music genre classification

##### Envelope
소리의 특성을 시간과 소리의 크기의 변화로 나타낸 그래프이며 네가지로 구성됨
- Attack: 시작 지점부터 소리의 크기가 최대인 부분
- Decay: 최대인 지점부터 중간크기로 떨어지는 부분
- Sustain: 중간크기로 지속되는 부분
- Release: 소리의 크기가 떨어져서 사라지는 부분

ex)드럼은 어택과 디케이가 매우 짧고, 바이올린은 어택이 느리고 서스테인이 길다.

### 이산 신호(離散信號)
연속신호를 샘플링한 신호이다. 연속 신호는 연속함수인 반면, 이산 신호는 수열이다. 이때 수열의 각 값을 샘플이라고 한다.
디지털 신호는 양자화된 신호이므로, 이산 신호는 디지털 신호와 구분되어야 한다. 다시 말하면, 이산 신호는 무한한 정밀도를 가지는 반면 디지털 신호는 8비트·16비트처럼 유한한 정밀도를 가진다.
<img src="images/speech/이산신호.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 
<img src="images/speech/디지털신호.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 

### Time domain  
시간을 기준으로 아날로그 시그널을 쪼개게 되는 것을 의미한다. Sampling을 통하여 컴퓨터는 소리 sequence를 binary value로 받아들이게 된다.

## Resampling
샘플링된 데이터를 다시금 더 높은 sampling rate 혹은 더 낮은 sampling rate로 다시 샘플링할수 있다. 이때는 일반적으로 interpolation(보간)을 할때는 low-pass filter를 사용한다.(Windowed sinc function)


## Spectral Envelope
Formants는 소리의 특징을 유추할 수 있는 중요한 단서가 된다. 우리는 포먼트들을 연결한 곡선과 Spectrum을 분리해야 한다. 그 곡선을 Spectral Envelope라 하고, MFCC는 이 분리 과정에서 도출 된다. 이때 사용하는 수학,알고리즘은 log와 IFFT(Inver FFT,역 고속 푸리에 변환)이다.


## Spectrum vs Spectrogram
공통점: 둘 다 소리의 주파수 및 강도를 분석해, 소리를 시각화 해주는 것이다

차이점:
1. Spectrum: 다양한 성분음들의 주파수와 진폭을 표시한 것을 그 소리의 스펙트럼이라고 한다. 스펙트럼을 통해 소리의 파형을 분석하여 어떠한 성분들이 이 소리를 구성하는가를 알 수있다.
(x축: 주파수, y축: 진폭(강도))

2. Spectrogram: 시간에 따른 각 주파수의 음향 에너지를 시각화 한것. 즉 스펙트로 그램은 스펙트럼에 시간의 축을 추가해, 주파수의 진폭의 시간에 따른 변화를 보여주는 3차원적인 그림.
(x축: 시간, y축: 주파수, z축: 진폭(강도))

<!-- Reference
https://ahnjg.tistory.com/47?category=1109653
[DMQA Open Seminar] 

Analysis of Sound data

https://hyunlee103.tistory.com/48
-->


# Trigonometric Form
모든 주기함수를 삼각함수들의 합으로 나타낼 수 있다. 

# Compact Form
sin 함수는 간단히 cos 함수로 변환할 수 있으니 cos 함수만의 합으로 나타낼 수도 있다. 

# Exponential Form
삼각함수는 즉 지수함수이니, 지수함수들의 합으로 나타낼 수도 있다.





### RNN(Recurrent Neural Network)
순차 데이터(Sequential data)를 다룰 수 있게 설계된 신경망이 순환 신경망(RNN : Recurrent Neural Network).
주가예측, AI작곡/작사, 기계번역, 음성인식 등 순차 데이터를 다루는 문제가 RNN의 주요 과제.
RNN은 과거의 정보를 현재에 반영해 학습하도록 설계. 이 컨셉을 통해 시간 순서로 나열된 데이터를 학습.
이전 상태로부터 전달된 값을 히든 스테이트(hidden state)라고 하고 이는 현재의 영향을 받아 매번 갱신.
활성화 함수에 tanh 이유? (비선형?) -> 품사 분류기? ->
RNN에서 주로 tanh를 사용하는 이유는,RNN의 Vanishing gradient 문제를 예방하기 위해서 gradient가 최대한 오래 유지될 수 있도록 해주는 역할로 tanh가 적합하기 때문입니다.
Sigmoid의 미분의 최대값이 0.25이기 때문에, Deep해질수록 Vanishing Gradient 가 발생합니다. 
Vanishing 문제를 해결하기 위해 만든 것이 Tanh 입니다. 
Tanh는 0~1인 Sigmoid 확장판이며, 미분의 최대값이 1로 Vanishing Graidient를 해결합니다.

1)tagging :output이 중요
2)Sentiment Analysis: 마지막 state value가 중요

RNN문제점
학습 데이터의 길이가 길어질수록 먼 과거의 정보를 현재에 전달하기 힘들기 때문인데요.
역전파 도중, 과거로 올라가면 올라갈수록 gradient 값이 소실(0에 수렴)되는 Vanishing gradient 문제발생

### LSTM(Long Short-Term Memory): 
RNN이 처리하지 못하는 장기 의존성(Long-Term dependencies) 문제를 다루기 위해 고안.
I grew up in France... I speak fluent French"라는 문단의 마지막 단어를 맞추고 싶다고 생각해보자. 최근 몇몇 단어를 봤을 때 아마도 언어에 대한 단어가 와야 될 것이라 생각할 수는 있지만, 어떤 나라 언어인지 알기 위해서는 프랑스에 대한 문맥을 훨씬 뒤에서 찾아봐야 한다. 이렇게 되면 필요한 정보를 얻기 위한 시간 격차는 굉장히 커지게 된다.
memory cell(어떠한 정보는 잊고, 어떠한 정보는 기억)
<!-- https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr -->


### Seq2seq(Sequence to Sequence)
seq2seq 는 번역기에서 대표적으로 사용되는 모델.
Encoder + Decoder로 구성.
1. 인코더(Encoder)는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 마지막에 이 모든 단어 정보들을 압축해서 하나의 벡터로 만드는데 이를 컨텍스트 벡터(Context vector)라고 한다. 
2. 입력 문장의 정보가 하나의 컨텍스트 벡터로 모두 압축되면 인코더는 벡터를 디코더로 전송한다.
3. 디코더(Decoder)는 컨텍스트 벡터를 받아서 번역된 단어를 한 개씩 순차적으로 출력한다.


### Attention
seq2seq 모델은 인코더에서 입력 시퀀스를 컨텍스트 벡터라는 하나의 고정된 크기의 벡터 표현으로 압축하고, 디코더는 이 컨텍스트 벡터를 통해서 출력 시퀀스를 만들어냈습니다. 하지만 이러한 RNN에 기반한 seq2seq 모델에는 크게 두 가지 문제가 있습니다. 
하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생합니다.
RNN의 고질적인 문제인 기울기 소실 (Vanishing Gradient) 문제가 존재합니다.
즉, 결국 이는 기계 번역 분야에서 입력 문장이 길면 번역 품질이 떨어지는 현상으로 나타났습니다. 이를 위한 대안으로 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정해주기 위한 등장한 기법이 어텐션(attention)입니다.
어텐션의 기본 아이디어는 디코더에서 출력 단어를 예측하는 매 시점(time step) 마다, 인코더에서의 전체 입력 문장을 다시 한 번 참고한다는 점입니다.
전체 입력 문장을 전부 다 동일한 비율로 참고하는 것이 아니라 해당 시점에서 예측해야할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중(attention)해서 보게 됩니다.


### Teacher forcing
잘못된 값이 나올 수 있기 때문에, 프리딕션 값대신 정답을 input 값으로 너어줌
<!-- 
reference: https://blog.naver.com/sooftware/221790750668
-->

<!-- 다시찾아 보기 -->
### Transformer 
RNN을 사용하지 않고 병렬처리 성능이 좋음.
Positional encoding으로 위치정보를 설정할때 사용(sin, cos)
장점 -1~1사이 값이 나옴, 긴 문장이 와도 가능 
self attention: query key value (이 세개는 벡터형태)


### end2end
모델의 모든 매개변수가 하나의 손실함수에 대해 동시에 훈련되는 경로가 가능한 네트워크로써 역전파 알고리즘 (Backpropagation Algorithm) 과 함께 최적화 될 수 있다는 의미이다.
예를들어 인코더(언어의 입력을 벡터로)와 디코더(벡터의 입력을 언어로)에서 모두가 동시에 학습되는 기계 번역 문제에서 효과적으로 적용 될 수 있다. 즉, 신경망은 한쪽 끝에서 입력을 받아들이고 다른 쪽 끝에서 출력을 생성하는데, 입력 및 출력을 직접 고려하여 네트워크 가중치를 최적화 하는 학습을 종단 간 학습(End-to-end Learning) 이라고 한다. 

Convolutional neural network 가 카메라의 원시 픽셀을 명령어에 직접 매핑하는 과정을 거치게 될 때 역전파는 종종 입력을 해당 출력으로 매핑하는 것과 관련하여 네트워크 가중치를 학습하는 효율적인 방법으로 사용된다. 만약, 신경망에 너무 많은 계층의 노드가 있거나 메모리가 적합하지 않을 경우 종단 간(end-to-end) 방식으로 훈련 시킬 수 없다. 이 경우 네트워크를 더 작은 네트워크의 파이프라인으로 나누어 해결 할 수 있는데 각 작은 네트워크는 독립적으로 훈련을 하게 되고 원하는 출력을 얻기 위해 연결 될 수 있다. 이러한 방식을 Divide and train 이라고 한다. 최적화가 중간 산출물에 의존한다는 점에서 국지적으로 수행이 되기 때문에 최적의 결과를 보장할 수 없다. 
<!-- 출처: https://eehoeskrap.tistory.com/183 [Enough is not enough:티스토리] -->

### word2vec
one hot encoding은 유사도가 없다는 문제점이 있음. 따라서 Embedding을 알아야함 (one hot인코딩보다 차원이 낮고, 유사도 분포알 수 있음)
word2vec는 임베딩 중에 하나임. ex_skipgram





<!--
referehce: 
DSBA [Paper Review] WaveNet: A generative model for raw audio
 -->