---
layout: post
title: "Digital signal processing"
image: speech1.jpg #12.jpg
date: 2022-07-02 19:59:18 -0500
tags: [ai, sound-ai]
categories: [sound-ai]
---
# Digital Signal Processing
### Audio 에서 우리가 얻을수 있는 정보는 크게 3가지이다.
1. Phase(Degress of displacement) : 위상
2. Amplitude(Intensity) : 진폭
3. Frequency : 주파수   
 *Frequency는 Hz 단위를 사용하며, 1초에 100번 period(주기)가 발생하는, 즉 100번 진동하는 소리를 100Hz로 정의한다. 그래서 f = 1/T 라는 수식이 성립한다.

(추가) 주기적힌 파형을 정현파 라고 부르는데, sin 곡선의 모양을 유지하면서, 일정한 속도로 진행하는 파형이라고 볼 수 있다.

# Sound Representation
1. 시간의 흐름에 따라, 공기의 파동의 크기로 보는 Time-domain Representation 
2. 주파수 영역에 따라 음압을 표현하는 Frequency-pressure Representation

<!-- ## 1.Time domain -->
## Waveform(파동)   
소리는 일반적으로 진동으로 인한 공기의 압축으로 생성되며, 압축이 얼마나 되었느냐에 따라서 표현되는 것이 Wave이다. Wave는 진동하며 공간/매질을 전파해 나가는 현상이다. 예를 들어 목소리의 경우 공기 분자가 진동을 하면서 발생한다. 즉 매질인 공기 분자가 얼마나 크게 흔들렸는지에 따라 형성되는 이러한 공기압의 진폭이 waveform 형태를 띄게 되어 우리가 보는 그래프가 그려진다. Y축은 Amplitude(진폭), X축은 Time(sec) 이다.

<img src="/assets/images/Speech/wave.jpg" width="90%" height="90%" title="waveform" alt="img"/> 


# ADC(Analog digital conversion)
```오디오 데이터는 연속형 데이터이다. 이를 딥러닝에 input으로 넣기 위해선 discrete한 벡터로 만들어야 한다. 이를 위해, Analog digital conversion 과정을 거쳐야하고, 이는 Sampling과 Quantization 두 step으로 이루어진다.```

## Sampling 
아날로그 정보를 잘게 쪼개서 discrete한 디지털 정보로 표현해야 하는데, 어떤 기준을 가지고 아날로그 정보를 쪼개서 대표값을 취하게 된다.

### Sampling rate
초당 sample 갯수를 의미한다.
예를 들어, Sample rate = 44100Hz인 소리의 경우 1초에 44100개의 sample을 뽑았다는 말이다. 
잘개 쪼갤수록 원본 데이터와 거이 가까워지기 떄문에 좋지만, Data의 양이 증가하게 되어,만약 너무 크게 쪼개게 된다면, 원본 데이터로 reconstruct하기 힘들어 진다는 문제가 있다.

Sample rate와 관련 된 법칙으로 Nyquist law가 있다. 모든 신호가 그 신호에 포함된 최고 주파수의 2배에 해당하는 빈도를 가지고 일정한 간격으로 샘플링하면 원래의 신호를 완벽하게 기록할 수 있다는 법칙이다. 
사람의 가청 주파수(20Hz ~ 20KHz)의 최고 주파수인 20KHz의 2배인 40KHz에 오차 허용 범위 10% 및 영상 업계 표준과의 동기화 문제 등으로 인해 대부분의 오디오 sample rate는 44100Hz 값을 갖게 된다.

샘플링 레이트가 최대 frequency의 2배 보다 커져야 한다는 것이다.

$$ f_{s} > 2f_{m} $$ 여기서 $$ f_{s} $$는 sampling rate이고, 
$$f_{m}$$ 은 maximum frequency를 말한다. 

Nyqusit frequency = $$ f_{s}/2 $$ sampling rate의 절반이다.

일반적으로 Sampling은 인간의 청각 영역에 맞게 형성 된다.
- Audio CD : 44.1 kHz(44100 sample/second)  
- Speech communication : 8 kHz(8000 sample/second)

*샘플링주기 구하기  
ex)1초당 10000번 표본 추출  
T = $$ \frac1{10,000} =0.1ms $$

*frame 구하기  
$$ d_f = \frac1{s_r} · {K} $$  
ex) $$ {s_r} $$ =44100, K=512 일 경우 11.6ms 

##  Quantization(양자화)  
표본화에서 얻어진 수치를 대표 값으로 n개의 레벨(discreate value)로 분해하고, 샘플 값을 근사 시키는 과정이다. 시간의 기준이 아닌 실제 amplitude의 real valued 를 기준으로 시그널의 값을 조절한다. Amplitude를 이산적인 구간으로 나누고, signal 데이터의 Amplitude를 반올림한다.
이산적인 구간은 bit의 비트에 의해서 결정된다.     
B bit의 Quantization : $$ -{2^{B-1}} $$ ~ $$ {2^{B-1}-1} $$  
Audio CD의 Quantization (16 bits) : $$ -2^{15} $$ ~ $$ 2^{15}-1 $$  
위 값들은 보통 -1.0 ~ 1.0  영역으로 scaling 되기도 한다.
즉, 비트로 표현 할 수 있게 애매한 값들을 비트로 표현 할 수 있는 값으로 근사 시키는 것이다. 
이러한 양자화 과정에서 샘플링 신호를 통해 반올림하거나 끝수를 버림으로써 근사치를 구하는데, 이 때, AD converter의 bit 수가 높으면 샘플링 된 신호의 버리는 수들이 적어져 본래 신호를 더 잘 살릴 수 있다.

이와 관련된 개념으로 Bit depth가 있다. Bit depth는 quantization을 얼마나 세밀하게 할지에 대한 정도로, 예를 들어 오디오 파일의 Bit depth = 16bits 이면, 16 비트(약 65536 levels)의 값으로 dicrete하게 양자화 된 소리임을 의미한다. 

## 부호화
양자화로 근사 시켜 bit로 표현 할 수 있게 되었지만, 양자화를 한 것은 수치를 근사화 만들면서 비트 값으로 표현 할 수 있게 밑거름을 만들어 준 것이지, 비트로 변환 된 것이 아니다. 때문에 "부호화" 과정을 통해 디지털 비트로 변환 해야 한다.
인코딩을 거쳐 '0'과 '1' 이진 비트로 표현된다. 대체적으로 사운드 파일의 원본의 용량은 크기 때문에 일반적으로 부호화 과정에서 압축하여 저장한다. 저장방식은 PCM(Pulse Coded Modulation), ADPCM(Adaptive Differential Pulse Coded Modulation)이 있다.
- PCM: 입력된 값 그대로를 기록하는 방식으로, 압축을 하지 않기 때문에 용량이 크며 CD 등에서 활용되는 고품질의 저장 방식으로 N개의 비트 한 세트를 PCM word라고 한다.


## Windowing(framing)
인풋 데이터(오디오)는 Sequential하고, time dependent하다. 따라서 time invariant(stationary)가 가능해지는 짧은 구간으로 신호를 쪼개는데, 이 과정을 windowing이라고 한다.

## Window Function
Windowing시 구간의 경계(양 끝 값)이 불연속해져서 끊기게 되어, 실제 신호와 달라지는 문제가 생기는데, 이렇게 windowing이 신호 특성에 주는 영향을 최소화 하기 위해 양 경계값을 0으로 수렴시키기는 window function을 각 구간(frame)마다 곱해준다.

## Overlapping frames
windowing시 발생할 수 있는 문제가 있다. 0으로 수렴시키는 구간의 signal이 사라진다는 점! 이부분을 해결 하기 위해 Overlapping frames를 사용한다. 이때 사용되는 개념: frame size = K, hop length


## Amplitude envelope
- Max amplitude value of all samples in a frame
- Gives rough idea of loudness
- Sensitive to outliers
- Onset detection, music genre classification

$$AE_t = {(t+1)·K-1} \quad {max} \quad k=t·K s(k)$$

## RMSE(Root-mean-square-energy)  
제곱평균제곱근 혹은 이차평균(quadratic mean)은 변화하는 값의 크기에 대한 통계적 척도이다. 이것은 특히 사인함수처럼 변수들이 음과 양을 오고 갈 때에 유용하다. Energy는 waveform이 가지고 있는 에너지 값을 의미한다. 즉 signal의 전체 amplitutde에 대응되는 값이다. RMSE는 나중에 MFCC feature중 하나로 사용된다.  
- RMS of all samples in a frame
- Indicator of loudness
- Less sensitive to outliers than AE
- Audio segmentation, music genre classification

$ RMS_t = \sqrt{\frac{1}{K}· \displaystyle\sum_{k=t·K}^{(t+1)·K-1} s(k)^2}$ = $\sqrt {Mean\,of\,sum\,of\,energy} $

<img src="/assets/images/Speech/AE+RMSE.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 


## Zero crossing rate  
(영교차율) 말 그대로 신호가 0을 지나는, 즉 신호의 부호가 바뀌는 비율을 말한다.  
- Recognition of percussive vs pithced sounds
- Monophonic pitch estimation
- Voice/unvioced decision for speech signals

$ZCR_t = \frac{1}{2}  \displaystyle\sum_{k=t·K}^{(t+1)·K-1}{|(sgn(s(k))-sgn(s(k+1))|}$

식을 보면 가장 먼저 현재 sample의 신호값과 바로 그 앞 sample의 신호값을 곱했을 때 이 값이 음인지 판단한다.부호가 바뀌었다면 그 결과는 음의 값이다. 그렇다면 |  | 안에 있는 식은 true가 되어 1을 반환하고,신호값을 곱한 결과가 0보다 크거나 같게 된다면 0을 반환한다.
신호의 길이만큼 그 연산을 하여 더하기를 하면 결과적으로 0을 교차한 횟수를 구할 수 있을 것이고, 신호의 길이만큼 교차횟수를 나누면 이는 신호가 영을 교차하는 비율이 된다.

<img src="/assets/images/Speech/Zero_crossing.svg.jpg" width="90%" height="90%" title="zcr" alt="img"/> 

# 푸리에 변환 (Fourier transform)
푸리에 변환(Fourier transform)을 직관적으로 설명하면 푸리에 변환은 임의의 입력 신호를 다양한 주파수를 갖는 주기함수(복수 지수함수)들의 합으로 분해하여 표현하는 것 입니다. 그리고 각 주기함수들의 진폭을 구하는 과정을 퓨리에 변환이라고 합니다.

- 주기(period): 파동이 한번 진동하는데 걸리는 시간, 또는 그 길이, 일반적으로 sin함수의 주기는 $2\pi /w$입니다
- 주파수(frequency): 1초동안의 진동횟수입니다.

퓨리에 변환의 식을 살펴봅시다.

$
y(t)=\sum_{k=-\infty}^\infty A_k \, \exp \left( i\cdot 2\pi\frac{k}{T} t \right)
$

이 식을 하나식 해석해봅시다. $k$는 $-\infty ~ \infty$의 범위를 가지고 움직입니다. 이것은 주기함수들의 갯수입니다. 어떠한 신호가 다른 주기함수들의 합으로 표현되는데, 그 주기함수는 무한대의 범위에 있군요.

그렇다면 $A_k$은 그 사인함수의 진폭이라고 합니다. 이 식은 시간에 대한 입력신호 $y_{t}$가  $\exp \left( i\cdot 2\pi\frac{k}{T} t \right)$와 진폭($A_k$)의 선형결합으로 표현됨을 말하고 있군요.


위 그림을 본다면 조금 더 명확히 알수 있을 것 같습니다. 붉은색 라인이 입력신호 $y_{t}$ 입니다. 일반적으로 우리가 다루게 되는 데이터인 음악이나 목소리 같은 데이터 역시 complex tone입니다. 여려개의 주파수영역이 합쳐진 것이죠. 이러한 여러개의 주파수 영역을 분리하자!가 주요한 아이디어입니다. 파란색 주기함수들을 보신다면 여러개의 주기함수들을 찾으실 수 있습니다. 그 주기함수들은 고유의 주파수(frequency)와 강도(amplitude)를 가지고 있고 그것이 파란색의 라인들로 표현되어 있습니다.

진폭에 대한 수식은 다음과 같습니다.

$$
A_k = \frac{1}{T} \int_{-\frac{T}{2}}^\frac{T}{2} f(t) \, \exp \left( -i\cdot 2\pi \frac{k}{T} t \right) \, dt
$$

여기서 하나의 의문점이 드실것 같습니다. 주기함수의 합으로 표현된다고 했는데 저희가 보고 있는것은 $\exp \left( i\cdot 2\pi\frac{k}{T} t \right)$ 지수함수의 형태이기 때문입니다.

지수함수와 주기함수 사이의 연관관계는 무엇일까요? 그 관계를 찾은 것이 바로 오일러 공식입니다.

$$
e^{i\theta} = \cos{\theta} + i\sin{\theta}
$$

이 식을 위 식처럼 표현한다면 다음과 같습니다
$$
\exp \left( i\cdot 2\pi\frac{k}{T} t \right) = \cos\left({2\pi\frac{k}{T}}\right) + i\sin\left({2\pi\frac{k}{T}}\right)
$$

여기서 $\cos{2\pi\frac{k}{T}}$, $i\sin{2\pi\frac{k}{T}}$ 함수는 주기와 주파수를 가지는 주기함수입니다. 

즉 퓨리에 변환은 입력 singal이 어떤것인지 상관없이 sin, cos과 같은 주기함수들의 합으로 항상 분해 가능하다는 것입니다. 


## DFT (Discrete Fourier Transform)

한가지 의문점이 듭니다. 바로, 우리가 sampling으로 들어온 데이터는 바로 시간의 간격에 따른 소리의 amplitude의 discrete한 데이터이기 때문이다. 그렇다면 위 푸리에 변환 식을 Discrete한 영역으로 생각해봅시다.

만약에 우리가 수집한 데이터 $$ y_{n} $$에서, 이산 시계열 데이터가 주기 $$ N $$으로 반복한다고 할때, DFT는 주파수와 진폭이 다른 $N$개의 사인 함수의 합으로 표현이 가능합니다.
$$
y_n = \frac{1}{N} \sum_{k=0}^{N-1} Y_k \cdot \exp \left( i\cdot 2\pi\frac{k}{N} n \right)
$$

위 식을 보면 k의 range가 0부터 $N-1$로 변화했음을 알 수 있다. 이때 Spectrum $Y_{k}$를 원래의 시계열 데이터에 대한 퓨리에 변환값이라고 하죠.

$$
Y_k = \sum_{n=0}^{N-1} y_n\cdot \exp \left( -i\cdot 2\pi\frac{k}{N} n \right)
$$

- $$ y_{n} $$ : input signal
- $$ n $$ : Discrete time index
- $$ k $$ : discrete frequency index
- $$ Y_{k} $$ : k번째 frequeny에 대한 Spectrum의 값


## Fourier Transform의 Orthogonal

$$
y(t)=\sum_{k=-\infty}^\infty A_k \, \exp \left( i\cdot 2\pi\frac{k}{T} t \right)
$$

어떠한 주기함수를 우리는 cos과 sin함수로 표현하게 되었습니다. 여기서 한가지 재밌는 점은, 이 함수들이 직교하는 함수(orthogonal)라는 점이다.
$$
\{ \exp \left(i\cdot 2\pi\frac{k}{T} t\right) \} = orthogonal
$$

벡터의 직교는 해당 벡터를 통해 평면의 모든 좌표를 표현할수 있었다. 함수의 내적은 적분으로 표현할 수 있는데, 만약 구간 [a,b]에서 직교하는 함수는 구간 [a,b]의 모든 함수를 표현할수 있습니다.

위 케이스에서는 cos, sin 함수가 사실상 우리 입력신호에 대해서 기저가 되어주는 함수라고 생각할 수 있습니다.

## FFT 

## STFT
푸리에 변환을 하면 time domain이 사라져 해석에 어려움이 생긴다. STFT는 frame마다 푸리에 변환을 취해 각각을 시간 순서로 옆으로 쌓아 time domain을 살리기 위한 방법이다.

---------
# 2.Frequency domain - Spectrum 
Spectrogram을 추출하는 일반적으로 프로세스는 입력신호에 대해서 window function을 통과하여 window size만큼 sampling 된 data를 받아서, Discrete Fourier Transform을 거치게 됩니다. DFT를 거친 신호들은 Frequency와 Amplitude의 영역을 가지는 Spectrum이 됩니다. 이후 이를 90도로 회전시켜서, time domain으로 stack하게 됩니다.

### Spectrogram
Spectrogram은 Frequency Scale에 대해서 Scaling이 이루어집니다. 주파수 영역에 Scaling을 하는 이유는, 인간의 주파수를 인식하는 방식과 연관이 있습니다. 
일반적으로 사람은, 인접한 주파수를 크게 구별하지 못합니다. 그 이유는 우리의 인지기관이 categorical한 구분을 하기 때문입니다. 때문에 우리는 주파수들의 Bin의 그룹을 만들고 이들을 합하는 방식으로, 주파수 영역에서 얼마만큼의 에너지가 있는지를 찾아볼 것입니다. 일반적으로는 인간이 적은 주파수에 더 풍부한 정보를 사용하기때문에, 주파수가 올라갈수록 필터의 폭이 높아지면서 고주파는 거의 고려를 안하게 됩니다.

따라서 아래 frequency scale은 어떤 방식을 통해 저주파수대 영역을 고려할 것이가에 대한 고민이 남아 있습니다.

### Linear frequency scale
일반적으로 single tone(순음)들의 배음 구조를 파악하기 좋습니다. 하지만 분포가 저주파수 영역에 기울어져(skewed) 있습니다.

### Mel Scale
사람의 청각기관은 High frequency 보다 low frequency 대역에서 더 민감하다. 사람의 이런 특성을 반영하여 물리적인 주파수와 실제 사람이 인식하는 주파수의 관계를 표연한 것이 Mel Scale 이다.
멜 스펙트럼은 주파수 단위를 다음 공식에 따라 멜 단위로 바꾼 것을 의미합니다.
$$
m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$
일반적으로는 mel-scaled bin을 FFT size보다 조금더 작게 만드는게 일반적입니다.

### Mel Spectrum
Mel Scale에 기반한 Filter Bank를 Spectrum에 적용하여 도출해 낸것이 Mel Spectrum이다. Mel Scale은 Filter Bank를 나눌때 어떤 간격으로 나누어야 하는지 알려주는 역할을 한다.

### MFCC(Mel-Frequenct Cepstral Coefficent)
MFCC는 오디오 신호에서 추출할 수 있는 feature로, 소리의 고유한 특징을 나타내는 수치.
기술적으로 말하자면, MFCC는 Mel Spectrum에서 Cepstral분석을 통해 추출된 값이다.

### Cepstral Analysis

Speech spectrum에서 Frequency domain에서 어떤 frequency가 활성화되어있는지 볼 수 있는데, 어떻게 envelope를 형성하느냐를 궁금해하고 찾고 싶어한다.

1. Audio Signal에 FFT를 수행하면 Spectrum이 된다
2. Spectrum에 Mel-Filter bank를 거치면 Mel-Spectrum이 된다.
3. -logX[k]를 log(Mel-Spectrum)이라고 할때, log X[k]에 Cepstral Analysis를 하면 log X[k] = log H[k] + log E[k]로 표현되고, 이것에 IFFT를 수행하면 x[k] = h[k] + e[k]가 된다.
4. h[k]를 Mel-Spectrum에 대해 얻은 Cepstral Coefficients(Cepstral계수)라고 하며, MFCC(Mel-Frequency Cepstral Coefficent)라고 부른다.

MFCC에서는 SFST를 통해 spectrum 뽑았다 -> mel-filters-> melspctrum -> log(melspectru)
-logX[k] = log H[k] + log E[k]

## Spectrogram
Windowing 과정에서 다뤘던 데이터에 대해 STFT로 power-spectrum을 구해 여기에 데시벨 변환 공식을 취해서 log-spectrum을 구하고 이를 세로로 세워서 frame마다 차곡차곡 옆으로 쌓으면, 푸리에 변환으로 사라졌던 time domain을 복원할 수 있고 이를 Spectrogram이라고 한다.

----
## 0.0.0 Baisc concept
##### Mono(Monophonic sound): 하나의 마이크 혹은 스피커와 같이 1개의 채널을 통해 듣는 소리.

##### Stero(Sterophonic sound): 스피커의 대칭 구성을 통해, 둘 이상의 독립 음향 채널을 사용하는 음향 재생 방식. streo로 구성된 오디오는 왼쪽,오른쪽 이 두개의 채널의 소리가 미세하게 다르게되어,소리가 공간감 있고 풍부하게 들린다.

##### Speaker Channel: 소리가 나오는 스피커가 몇개인지를 말하며, 채널이 많으면 소리가 여러 스피커를 통해 분산해 나온다고 생각하면 된다.

##### Harmonics(배음)구조
소리는 한 가지의 주파수만으로 구성되지 않기 때문에, 기본 주파수(Fundamental frequency)와 함께 기본 주파수의 정수배인 배음(harmonics)들로 구성된다. 
예를들어 피아노 건반에서 4옥타브 '라'(440Hz)음을 연주했다면 그소리는 기본주파수인 440Hz뿐만 아니라, 그 정수배인 880Hz, 그리고 그 다음 배음들까지 포함한다.

##### Filter Bank
??

##### Formants(포먼트)
소리가 공명되는 특정 주파수 대역이다.
사람의 음성은 Vocal folds(성대)애서 형성되어, Vocal track(성도)를 거치며 변형되는데, 소리는 성도를 지날때, 포먼트를 만나 증폭되거나 감쇠된다. 즉. Formant(포먼트)는 Harmonics 배음과 만나 소리를 풍성하게 혹은 선명하게 만드는 필터역할을 한다.


##### 정현파 (Sinusoid)
모든 신호는 주파수(frequency)와 크기(magnitude), 위상(phase)이 다른 정현파(sinusolida signal)의 조합으로 나타낼 수 있다. 퓨리에 변환은 조합된 정현파의 합(하모니) 신호에서 그 신호를 구성하는 정현파(주기적인 파형)들을 각각 분리해내는 방법이다.  
$$ 오일러 공식: e^jθ=cosθ+jsinθ$$ 


### Time domain  
시간을 기준으로 아날로그 시그널을 쪼개게 되는 것을 의미한다. Sampling을 통하여 컴퓨터는 소리 sequence를 binary value로 받아들이게 된다.

## Resampling
샘플링된 데이터를 다시금 더 높은 sampling rate 혹은 더 낮은 sampling rate로 다시 샘플링할수 있다. 이때는 일반적으로 interpolation(보간)을 할때는 low-pass filter를 사용한다.(Windowed sinc function)


## Spectral Envelope
Formants는 소리의 특징을 유추할 수 있는 중요한 단서가 된다. 우리는 포먼트들을 연결한 곡선과 Spectrum을 분리해야 한다. 그 곡선을 Spectral Envelope라 하고, MFCC는 이 분리 과정에서 도출 된다. 이때 사용하는 수학,알고리즘은 log와 IFFT(Inver FFT,역 고속 푸리에 변환)이다.


<!-- Reference
https://ahnjg.tistory.com/47?category=1109653
[DMQA Open Seminar] 

Analysis of Sound data

https://hyunlee103.tistory.com/48
-->


# Summary

1. Audio Signal에 FFT를 수행하면 Spectrum이 된다
2. Spectrum에 Mel-Filter bank를 거치면 Mel-Spectrum이 된다.
3. -logX[k]를 log(Mel-Spectrum)이라고 할때, log X[k]에 Cepstral Analysis를 하면 log X[k] = log H[k] + log E[k]로 표현되고, 이것에 IFFT를 수행하면 x[k] = h[k] + e[k]가 된다.
4. h[k]를 Mel-Spectrum에 대해 얻은 Cepstral Coefficients(Cepstral계수)라고 하며, MFCC(Mel-Frequency Cepstral Coefficent)라고 부른다.


*DFT : 음성 신호를 주파수영역으로 변환
*log: 빠른 신호와 느린 신호를 덧셈의 형태로 변환
*IDFT: 느린 신호는 time축의 낮은 영역에 존재, 빠른 신호는 time축의 높은 역역에 존재하게 만듬
        (시간축과 주파수 축의 duality 성질 때문에)


#### 음성신호처리에서 frame생성시 overlap 하는 이유
음성신호는 stationary하지 않지만, 그렇게 빠르게 변하지도 않는다. 이때 신호를 stationary한 작은 청크로 잘라서 처리하면 되겠다는 생극을 할 수 있다. Windowing은 신호의 어느부분을 볼지 정하는 창이다. 이르 이용해 신호를 작은 청크로 짜른다. 이 청크를 frame이라 한다. 그래서 충분히 작은 frame으로 자른다면, stationary한 신호를 얻을 수 있다.

이렇게 frame단위로 잘라서 처리를 하게되는데, 인접한 두 프레임을 처리할 때 두 프레임간의 property의 변화로 인해 프레임간의 불연속, 혹은 jump가 일어날 수 있다. 쉽게 받아들이기 위해 이미지를 예로 설명하자면,  8X8 non-overlapping blocks을 이용해서 압축을 한 JPEG(압축된 image)이미지에 tiling이나 blocking효과가 생겨서 화질이 안좋아진다. 심지어 듣는 것은 어떻겠는가...음질이 안좋아진다

frame을 생각해보면 중간에는 당연히 안정적인 구조가 나올 것이고 왼쪽/오른쪽 양 끝에서 가장 많이 다를 것이다. 이것이 이해가 안된다면, window의 모양을 생각해보면 된다. 우리가 FFT를 할 때 사용하는 윈도우들은 대체로 테이퍼드의 형태를 사용하고 있다. 이를테면 hamming window는 아래와 같은 형태를 띈다.

Time domain의 window를 보면, 가장자리에서 0에 가까운 값을 가짐을 알 수 있다. 이런 window를 써서 FFT를 하게 되면 transient같은 중요한 정보를 손실할 수 있다. 따라서 window를 사용하므로써 발생하는 부정적인 영향을 줄이기 위해 fft에서 overlap을 사용한다. 그래서 frame간의 이런 급작스러운 변화를 막고 이를 보정하기 위해 overlap을 쓴다고 보면 되겠다. 결론적으로 overlapping window는 frame의 양 끝단에서 신호의 정보가 자연스럽게 연결되게 하기 위해 사용한다.


<!--
Reference
-->