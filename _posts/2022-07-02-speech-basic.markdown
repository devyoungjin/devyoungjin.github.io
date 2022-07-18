---
layout: post
title: "Speech Basic"
image: speech1.jpg #12.jpg
date: 2022-07-02 12:00:18 +0200
tags: [ai, sound-ai]
categories: [ai, sound-ai]
---

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