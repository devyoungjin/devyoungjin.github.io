---
layout: post
title: "VAE"
image: speech1.jpg #code1.jpg 
date: 2022-07-05 00:00:18 +0200
tags: [ai, vae]
categories: [sound-ai]
---

# Transformer

2021년 기준으로 최신 고성능 모델들은 Transformer아키텍처를 기반으로 하고 있다


GPT: Transformer의 Decoder 아키텍처 활용
BERT: Transformer의 Encoder 아키텍처 활용

기계번역 발전 과정  
RNN(1986) -> LSTM(1997) -> Seq2Seq(NIPS2014) -> Attention(ICLR2015) -> Transformer(NIPS 2017) -> GPT(2018) -> BERT(NAACL2019)

*seqseq: 고정된 크기의 context vector사용  
*Attention이후 입력 시퀀스 전체에서 정보를 추출하는 방향으로 발전

## seq2seq 모델의 한계쩜
context vector 소스 문장 정보를 압축: 병목현상(bottleneck)이 발생하여 성능 하락의 원인이 됨. 즉 하나의 문맥 벡터가 소스 문장의 모든 정보를 가지고 있어야 하므로 성능이 저하됨. 

### 해결 방안
Seq2Seq모델에 Attention(디코더는 인코더의 모든 출력(outputs을 참고))
매번 소스 문장에서의 출력 전부를 입력으로 받으면 어떨까라는 아이디어에서 착안 (최신 GPU는 많은 메모리와 빠른 병렬 처리를 지원한다.)

### seq2seq with Attention:(Decoder)
디코더는 매번 인코더의 모든 출력 중에 어떤 정보가 중요한지를 계산한다

Energy  $e_{ij} = a(s_{i-1},h_j)$ : 어떤 h값과 가장 많은 연관성이 있는지 

Weight $ a_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})} $ : 소프트 맥스를 통해 가중치가 반영된 각 인코더의 값을 추후 더해서 활용

- i = 현재의 디코더가 처리중인 인덱스
- j = 각각의 인코더 출력 인덱스
- s: 디코더가 이전에 출력했던 단어를 만들기위해 사용했던 hidden state
- h: 인코더 파트의 각각의 hidden state

*Attention 가중치를 활용해 각 출력이어떤 입력정보를 참고했는지 알 수 있다는 장점이 있다.

---

# Transformer

2021년 기준 현대 자연어 처리 네트워크의 핵심 논문
RNN이나 CNN을 전혀 필요로 하지 않고 대신 Positional Encoding을 사용.  
인코더와 디코더로 구성되며, Attention과정을 여러번 레이어에서 반복한다.

## Transformer의 동작 원리: 입력 값 embedding

.트랜스 포머 이전의 전통적인 인 임베딩은, 입력차원 자체는 *Input Embedding Matrix는 단어의 갯수만큼 행의 크기를 가지고, 각각의 열 데이터는 임베딩 차원의 같은 크기(논문512)를 갖는다. 
RNN을 사용하지 않으려면 위치 정보를 포함하고 있는 임베딩을 사용해야 한다.
이를 위해 트랜스 포머에서는 *Positional Encoding을 사용한다. 즉 Attention 이 받는 값은 Matrix + 위치 정보가 포함된 값이다. 

성능 향상을 위해 잔여 학습(Residual Learning)을 사용한다. 이후 Normalization. 어텐션과 정규화 과정을 반복하는 방식으로 여러개의 레이러를 중첩해서 사용한다. 각 레이어는 서로 다른 파라미터를 가진다. 

트랜스포머에서는 마지막 인코더 레이어의 출력이 모든 디코더 레이어에 입력에 사용된다. 

## Transformer의 동작 원리: Attention
인코더와 디코더는 Multi-Head Attention레이어를 사용한다.

Scaled Dot-Product Attention 을 위한 세가지 입력 요소: Query,Key,value

Multi-Head Attention

Attention(Q,K,V) = softmax $ (\frac {QK^T}{\sqrt d_k})V $

$head_i$ = Attention $ (QW_i^Q, KW_i^K, VW_i^V) $
MultiHead(Q,K,V) = Concat $ (head_1, head_2, ..., head_h)W^O$
- Q: Query
- K: Key
- V: value
- h: head의 개수
- $d_k$ : key dimension

*Mast matrix를 이용해 특정 단어는 무시할 수 있다
: 마스크 값으로 음수 무한의 값을 넣어 softmax함수의 출력에 0%에 가까워 지도록 한다.

## Transformer의 동작 원리: Attention종류
트랜스포머에서는 세가지 종류의 Attention레이어가 사용된다
1. Encoder self-Attention: 인코더와 디코더 모두에서 사용돠며, 매번 입력 문장에서 각 단어가 다른 어떤 단어와 연관성이 높은지 계산할 수 있다.
2. Masked Decoder Self-Attention: 각 출력단어가 앞에 등장한 단어만 참고 하도록 만듬
3. Encoder-Decoder Attention: 쿼리가 디코더에 있고, 키 벨류가 인코더에 있는 상황 

## Transformer의 동작 원리: Positional Encoding
Positional Encoding은 다음과 같이 주기함수를 활용한 공식을 활용한다
각 단어의 상대적인 위치 정보를 네트워크에 입력한다
$ PE_(pos,2i) = sin(pos/10000^{2i/d_model})$  
$ PE_(pos,2i+i) = cos(pos/10000^{2i/d_model})$
- PE: positional encoding
- pos :각 단어 번호
- i : 각 단어에 대한 embedding값 위치