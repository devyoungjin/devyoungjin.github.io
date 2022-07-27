---
layout: post
title: "Digital signal processing"
image: speech1.jpg #12.jpg
date: 2022-07-02 19:59:18 -0500
tags: [ai, sound-ai]
categories: [sound-ai]
---
```
공부한 내용을 작성중입니다. 잘못된 정보에 대해서는 피드백 주시면 감사하겠습니다.
```
# Digital Signal Processing

## Audio 에서 얻을수 있는 정보 
1. Phase(Degress of displacement) : 위상
2. Amplitude(Intensity) : 진폭
3. Frequency : 주파수   
- Frequency는 Hz 단위를 사용하며, 1초에 100번 period(주기)가 발생할 경우(100번 진동하는 소리)를 100Hz로 정의
따라서f = 1/T 라는 수식이 성립한다  
## Waveform(파동)   
- 소리: 진동으로 인한 공기의 압축으로 생성한다  
- Wave: 진동하며 공간/매질을 전파해 나가는 현상, 소리가 압축이 얼마나 되었느냐에 따라서 표현되는 것이다

*Y축은 Amplitude(진폭): 매질인 공기 분자가 얼마나 크게 흔들렸는지에 따라 형성한다
*X축은 Time(sec)이다  

<img src="/images/speech/wave1.jpg" width="90%" height="90%" title="waveform" alt="img"/> 


# ADC(Analog digital conversion)
```오디오는 연속형 데이터이기 때문에 discrete한 벡터로 만들어야 한다. 이를 위해  Sampling과 Quantization 의 과정을 거쳐야 하며, 이를 Analog digital conversion라 한다.```  

1. Sampling: 아날로그 데이터 시간축에 대한 디지털화
2. Quantization: 아날로그 데이터 진폭축에 대한 디지털화

<img src="/images/speech/adc.jpg" width="90%" height="90%" title="waveform" alt="img"/> 


## Sampling 
아날로그 정보를 discrete한 디지털 정보로 표현하기위해 아날로그 정보를 쪼개서 대표값을 추출하는 것이다

## Sampling rate(sr)
초당 sample 갯수를 의미한다  
EX) Sample rate = 44100Hz인경우 1초에 44100개의 sample을 뽑았음을 의미   
잘개 쪼갤수록 좋지만, 너무 크게 쪼개게 되면, 효율성 및 연산 속도가 떨어지고  원본 데이터로 reconstruct하기 힘들어 진다.

```
Sampling rate와 관련 된 법칙으로 Nyquist law가 있다. 
```

샘플링된 신호는 이후 양자화를 시킨다. 양자화란 여러 단계로 나뉜 범위안에서 샘플링된 신호에 가까운 범위를 대표하는 정수값으로 바꾸는 것인데, 정수로 바뀌는 과정에서 양자화 오차가 발생한다.
이 오차 때문에, 신호를 복원했을 때, 우리는 기존 아날로그 신호와는 차이가 존재하는 신호를 얻게되는데, 신호를 복원했을 때 기존 아날로그 신호의 유실 없이 복원되기 위해서는 얼마만큼 신호를 샘플링해야 하는지 나이퀴스트 정리를 보고 판단하면 된다.


### Nyquist law

특정 주파수 대역(Hz)의 소리를 분석하려면, 샘플링 레이트(Hz)가 최소 2배 이상 되도록 샘플링 해야 한다는 법칙이다.  
$ f_{s} > 2f_{m} $ 여기서 $ f_{s} $는 sampling rate이고, 
$ f_{m}$ 은 maximum frequency를 말한다. 

$ \therefore $ Nyqusit frequency = $ f_{s}/2 $, 즉 sampling rate의 절반이다.

ex) sr이 44.1 kHz(44,100 Hz)라면 주파수 대역이 0~22,050 Hz 사이인 소리만 분석 할 수 있다는 것이고, 주파수 대역이 0~22,050 Hz 인 소리를 분석하려면, sr이 44.1 kHz(44,100 Hz)는 되어야 한다는 것이다.

```
일반적으로 Sampling은 인간의 청각 영역에 맞게 형성 된다
사람의 가청 주파수(20Hz ~ 20KHz)의 최고 주파수인 20KHz의 2배인 40KHz에 오차 허용 범위 10% 및 영상 업계 표준과의 동기화 문제 등으로 인해 대부분의 오디오 sample rate는 44100Hz 값을 갖게 된다.  
- Audio CD : 44.1 kHz(44100 sample/second)  
- Speech communication : 8 kHz(8000 sample/second)
```

* 샘플링 주기 구하기  
ex)1초당 10000번 표본 추출  
T = $ \frac1{10,000} =0.1ms $

* frame 구하기  
$ d_f = \frac1{s_r} · {K} $  
ex) $ {s_r} $ =44100, K=512 일 경우 11.6ms 

##  Quantization(양자화)  
표본화에서 얻어진 수치(continous)를 대표 값으로 n개의 레벨(discreate value)로 분해하고, 샘플 값을 근사 시키는 과정이다. 시간의 기준이 아닌 실제 amplitude의 real valued 를 기준으로 시그널의 값을 조절한다. Amplitude를 이산적인 구간으로 나누고, signal 데이터의 Amplitude를 반올림한다.

이와 관련된 개념으로 Bit depth가 있다. 위에서 말한 이산적인 구간을 결정하는 것이다.

### Bit dept
Bit depth는 quantization을 얼마나 세밀하게 할지에 대한 정도로, 이산적인 구간은 bit의 비트에 의해서 결정된다.       

B bit의 Quantization : $ -{2^{B-1}} $ ~ $ {2^{B-1}-1} $  
Audio CD의 Quantization (16 bits) : $ -2^{15} $ ~ $ 2^{15}-1 $  
위 값들은 보통 -1.0 ~ 1.0  영역으로 scaling 되기도 한다.
즉, 비트로 표현 할 수 있게 애매한 값들을 비트로 표현 할 수 있는 값으로 근사 시키는 것이다. 
이러한 양자화 과정에서 샘플링 신호를 통해 반올림하거나 끝수를 버림으로써 근사치를 구하는데, 이 때, AD converter의 bit 수가 높으면 샘플링 된 신호의 버리는 수들이 적어져 본래 신호를 더 잘 살릴 수 있다.

ex) 오디오 파일의 Bit depth = 16bits 이면, 16 비트(약 65536 levels)의 값으로 dicrete하게 양자화 된 소리임을 의미함 

```
양자화 잡음
A/D 변환 과정 중 양자화 시 나타나는 오차

*양자화 잡음을 줄이기 위해
1. 양자화 스텝 수 증가
2. 비선형 양자화 진행
3. 압축신장 방식 적용
```

### 부호화
샘플링,양자화를 거친 데이터를 유의미한 값의 데이터로 도출하는 과정이다.  
양자화로 근사 시켜 bit로 표현 할 수 있게 되었지만, 양자화를 한 것은 수치를 근사화 만들면서 비트 값으로 표현 할 수 있게 만들어 준 것이지, 디지털 비트로 변환된 것이 아니므로 인코딩을 거쳐 '0'과 '1' 이진 비트로 표현된다.  
```
대체적으로 사운드 파일의 원본의 용량은 크기 때문에 일반적으로 부호화 과정에서 압축하여 저장한다.   
저장방식은 PCM(Pulse Coded Modulation), ADPCM(Adaptive Differential Pulse Coded Modulation)이 있다.
```
---

# Feature Extraction
<img src="/images/speech/mfcc1.jpg" width="90%" height="90%" title="waveform" alt="img"/>     

오디오 데이터는 매우 고차원이고 여러 frequency 가 섞여 있기 때문에, feature를 뽑아내야 한다. 가장 대표 feature 로는 MFCC가 있다.

## windowing 와 frame
도잉(windowing)은 신호를 어느부분을 볼지 정하는 창이다. 이를 이용해 신호를 작은 청크로 자른다. 이 청크를 frame이라고 부르고, 작은 frame으로 적절하게 자른다면, stationary한 신호를 얻을 수 있다.


## Window Function
Windowing시 구간의 경계(양끝)이 불연속해져서 끊기게 되어, 실제 신호와 달라지는 문제가 생기는데, 이렇게 windowing이 신호 특성에 주는 영향을 최소화 하기 위해 양 경계값을 0으로 수렴시키기는 window function을 각 구간(frame)마다 곱해준다. 또한 frame간의 이런 급작스러운 변화를 막고 이를 보정하기 위해 overlap을 사용한다.
```
Haming window
```


# Energy
energy는 waveform이 가지고 있는 에너지 값 이다. 즉 signal의 전체 amplitude에 대응되는 값이다. signal의 각 amplitude 포인트를 $x_n$이라고 할 때, signal의 energy는 다음과 같이 정의된다.  
$\displaystyle\sum_{k=n} {|x(n)|}^2$

## RMSE(Root-mean-square-energy)  
제곱평균제곱근 혹은 이차평균(quadratic mean)은 변화하는 값의 크기에 대한 통계적 척도이다. 이것은 특히 사인함수처럼 변수들이 음과 양을 오고 갈 때에 유용하다. Energy는 waveform이 가지고 있는 에너지 값을 의미한다. 즉 signal의 전체 amplitutde에 대응되는 값이다. RMSE는 나중에 MFCC feature중 하나로 사용된다.  
- RMS of all samples in a frame
- Indicator of loudness
- Less sensitive to outliers than AE
- Audio segmentation, music genre classification

$ RMS_t = \sqrt{\frac{1}{K}· \displaystyle\sum_{k=t·K}^{(t+1)·K-1} s(k)^2}$ = $\sqrt {Mean\,of\,sum\,of\,energy} $

<img src="/assets/images/Speech/AE+RMSE.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 


## Zero crossing rate  
신호의 부호가 바뀌는(신호가 0을 오고가는) 비율을 말한다.  
- Recognition of percussive vs pithced sounds
- Monophonic pitch estimation
- Voice/unvioced decision for speech signals

$$ ZCR_t = \frac{1}{2}  \displaystyle\sum_{k=t·K}^{(t+1)·K-1}{|(sgn(s(k))-sgn(s(k+1))|} $$

식을 보면 가장 먼저 현재 sample의 신호값과 바로 그 앞 sample의 신호값을 곱했을 때 이 값이 음인지 판단하고, 부호가 바뀌었다면 그 결과는 음의 값이다. 그리고 |  | 안에 있는 식은 true가 되어 1을 반환하고,신호값을 곱한 결과가 0보다 크거나 같게 된다면 0을 반환한다.
신호의 길이만큼 그 연산을 하여 더하기를 하면 결과적으로 0을 교차한 횟수를 구할 수 있을 것이고, 신호의 길이만큼 교차횟수를 나누면 이는 신호가 영을 교차하는 비율이 된다.

<img src="/assets/images/Speech/Zero_crossing.svg.jpg" width="90%" height="90%" title="zcr" alt="img"/>   

---

## 푸리에 급수 vs 푸리에 변환
- 푸리에 급수: 시간에 따라 반복되는 주기 함수를 변환하는데, 주기 신호에만 사용할 수 있다는 단점이있다.   
- 푸리에 변환: 주기 신호에만 사용할 수 있다는 단점을 극복하기 위해 나온 개념으로 푸리에 변환은 주기성이 없는 비주기 함수를 주기가 무한대인 주기함수로 보고 푸리에 급수를 활용할 수 있다.  
푸리에 변환은 임의의 입력 신호를 다양한 주파수를 갖는 주기함수(복수 지수함수)들의 합으로 분해하여 표현하고 각 주기함수들의 진폭을 구하는 과정이다.

###  푸리에 변환이 필요한 이유?
푸리에 변환의 가장 큰 목적은, 시간 t축에서 존재하는 신호, 정보를 주파수f축으로 옮겨오는데 있다. 시간축에 존재하던 신호를 주파수로 옮겨오면 전송해야할 데이터의 양은 줄어든다.


# Fourier transform 유도

a = 가로, b = 세로, gamma = a와 b사이의 각도 라고 하면,  
c = a와 b를 이은 대각선 = 복소수(complex number)

$ γ(gamma) = arctan(\frac b a) $ 

$ |c| = \sqrt{a^2+b^2}  $

$ a=|c|\,·\,cos(γ)$    
$ b=|c| \,·\,sin(γ)$  
$ c = |c|\,·\, (cos(γ) + i sin(γ)$  

### Euler fomular(오일러 공식)
$ γ(gamma) = 2πθ$??  
오일러공식 $ e^{2πiθ} = cos(2πθ) + i sin(2πθ) $  
오일러공식 $ e^{iγ} = =cos(γ) + i sin(γ)$

### Euler identity(오일러 항등식)
$ e^{i\pi} + 1 =0$ 이기 때문에 $e^{i\pi}=-1$이고, 오일러공식 $ e^{iγ} = cos(γ) + i sin(γ)$ 에서 γ에 π를 대입하면 cos(π) = -1, i sin(π)=0 이고, $e^{iπ}$ = -1이다.

결국  
$ c = |c|\,·\, (cos(γ) + i sin(γ)$  
$ e^{iγ} = =cos(γ) + i sin(γ)$을 보면  
$c = |c|\,·\,e^{iγ}$가 된다.   
$e^{iγ}$는 Direction of number in the complex plane(복소평면의 숫자 방향) 이고 \, |c| 는 Scales distance from origin(원점으로부터의 거리)이다

# 푸리에 변환(Fourier transform)
$ c_f = \frac {d_f} {\sqrt 2} \,·\, e^{i2\pi φ_f} $ 이고, 이것을 $c = |c|\,·\,e^{iγ}$와 비교해보면, $2πφ_f$ 는 γ각도 이다.
$φ_f$은 0과 1사이 이다.  - 가 붙은 이유는 위상을 높이면, 시계 방향으로 돌기 위해서이다. 

## Complex Fourier transform  
$ĝ(f) = c_f$(계수 coefficent)  
$ĝ$: R(frequency)-> C(complex number)   
Fourier transform은 두개의 다른 매개변수(크기,위상)이 나오므로 complex plain에 표시한다.


## define of complex Fourier transform  
$ \hat g(f) = ∫g(t)\,·\,e^{-i2πft}$ dt 
- $e^{-i2πft}$:(pure torn) 지수는 복소평면에서 단위 원을 시계방향으로 추적한다. 원을 완성할 수 있는 속도는 f value (frequency)에 달려있다.  

 ex) frequency=1 (1Hz)는 원 한바퀴를 도는데 1초가 든다.  
    2초면 1/2초가 걸린다.(주기가 항상 주파수의 역수와 같기 때문에)     
- $\hat g(f)$: original signal

오일러공식 $ e^{iγ} =cos(γ) + i sin(γ)$
을 $e^{-i2πft}$ 에 대입하면, $ \hat g(f)$ = $ ∫g(t)\,·\,e^{-i2πft} $ dt = $ ∫g(t) ·\, cos(-2πft)$ dt + $i∫ g(t)·sin(-2πft)$ dt 이다.   
*공식을 더 간단하게 하기 위해 삼각함수가 아닌  $e^{i2πft}$로 사용하는 것이다.

- $ ∫g(t) ·\, cos(-2πft)dt$ = Real part   
- $ i ∫ g(t)·sin(-2πft)$ dt = Imaginary part이기 때문에  
$ \hat g(f) $ 는 복소수가 될것이고, complex Fourier transform의 계수이다.

다시 complex Fourier transform 식을 보면 $ c_f = \frac {d_f} {\sqrt 2} \,·\, e^{i2\pi φ_f} $ 이고, ${\sqrt 2}$는 정규화 상수이므로 크게 신경쓰지 않는다. 이를 바꾸면 $ d_f = {\sqrt 2} \,·\, |\hat g(f)|$이고, 위상측면에서는 ${φ_f}= -\frac {γ_f} {2π}$ 이고 γ=2π${φ_f}$ 와 같다.

---

## DFT (Discrete Fourier Transform)
이산 푸리에 변환(discrete Fourier transform, DFT)은 이산적인 입력 신호에 대한 푸리에 변환으로, 디지털 신호 분석과 같은 분야에 사용된다. 또한 이산 푸리에 변환은 고속 푸리에 변환을 이용해 빠르게 계산할 수 있다. 

*FFT (Fast Fourier Transform)는 DFT를 빠르게 계산할 수 있는 알고리즘으로 FFT의 결과값과 DFT의 결과값은 동일하지만 계산량을 대폭 줄일 수 있다.

퓨리에 변환은 무한대의 시간에 대해 적분하는 반면, DFT는 일정한 개수의 샘플을 샘플링 주기로 합하여 계산한다. DFT의 결과는 복소수 값으로 나오며 복소수 값으로 특정 주파수의 신호의 크기와 위상을 알 수 있다.


DFT:

$$ {X_n} = \sum_{n=0}^{N-1} x_k\cdot \exp \left( -i\cdot 2\pi\frac{kn}{N} \right)$$

- $ X_{n}$ : input signal
- $ n $ : Discrete time index
- $ k $ : discrete frequency index
- $ x_{k}$ : k번째 frequeny에 대한 Spectrum의 값

IDFT(inverse discrete Fourier transform)
$$ {x_k} = \frac{1}{N} \sum_{n=0}^{N-1} X_n\cdot \exp \left( -i\cdot 2\pi\frac{kn}{N} \right)$$


t에 대한 함수를 푸리에 변환하면 f에 대한 함수가 되고, f에 대한 함수를 푸리에 역변환하면 t에 대한 함수가 된다.

<!-- $X(w)= \int_{-\infty}^{\infty}x(t)e^{jwt}dt$  $\Leftrightarrow$ 
$x(t)= \frac{1}{2\pi}X(w)e^{jwt}dw $ -->


## DTFT(이산시간푸리에변환)
이산 시간 푸리에 변환은 무한 신호에 대한 변환이다.
무한하게 다양한 주파수(frequency)를 가진 정현파를 각각 얼마나 가중해서 더해주느냐에 따라 모든 복잡한 주기함수를 표현할 수 있음을 표현한다.
$$ y(t)=\sum_{k=-\infty}^\infty A_k \, \exp \left( i\cdot 2\pi\frac{k}{T} t \right) $$


 - y(t): 근사 또는 표현하고 싶은 복잡한 주기 함수
 - t: continuous variable
 - T: 함수f의 시간주기
 - $A_k$: 계수
 - $\exp \left( i\cdot 2\pi\frac{k}{T} t \right)$: 복잡한 함수를 구성하는 간단한 주기함수
 - n : 다른 주파수를 가진 sin, cos 을 나타내기 위한 정수배  
 - j: 허수 $\sqrt{-1}$

이 식을 하나식 해석해보면, $k$는 $-\infty ~ \infty$의 범위를 가지고 움직인다. 이것은 주기함수들의 갯수이다. 어떠한 신호가 다른 주기함수들의 합으로 표현되는데, 그 주기함수는 무한대의 범위있다고 생각하면 된다.

$A_k$은 그 사인함수의 진폭이라고 생각하면 된다. 이 식은 시간에 대한 입력신호 $ y_{t} $가  $\exp \left( i\cdot 2\pi\frac{k}{T} t \right)$와 진폭($A_k$)의 선형결합으로 표현됨을 말한다.

진폭에 대한 수식은 다음과 같다.

$A_k = \frac{1}{T} \int_{-\frac{T}{2}}^\frac{T}{2} f(t) \, \exp \left( -i\cdot 2\pi \frac{k}{T} t \right) \, dt$

 주기함수의 합으로 표현된다고 했는데  $$\exp \left( i\cdot 2\pi\frac{k}{T} t \right)$$ 는 지수함수의 형태이다. 여기서 우리는 지수함수와 주기함수 사이의 연관관계를 "오일러 공식"으로 확인해볼 수 있다.

$
{e^{i\theta}} = \cos{\theta} + i\sin{\theta}
$

이 식을 위 식처럼 표현한다면 다음과 같다
$
{\exp} \left( i\cdot 2\pi\frac{k}{T} t \right) = \cos\left({2\pi\frac{k}{T}}\right) + i\sin\left({2\pi\frac{k}{T}}\right)
$

여기서 $ \cos{2\pi\frac{k}{T}} $, $ i\sin{2\pi\frac{k}{T}} $ 함수는 주기와 주파수를 가지는 주기함수이다. 

즉 퓨리에 변환은 입력 singal이 어떤것인지 상관없이 sin, cos과 같은 주기함수들의 합으로 항상 분해 가능하다는 것이다 


# Fourier Transform의 Orthogonal(직교성)

<!-- reference: https://m.blog.naver.com/sagala_soske/220983389992 -->

### 삼각함수의 직교성  
- 삼각함수는 서로 직교하기 때문에 두 함수를 내적하면 0이 나와야 한다.  
벡터의 내적은 두 벡터 값으 곱이지만, 함수는 서로 곱하고 적분한다. 
이 때 적분 범위는 삼각함수의 주기(0,$ 2\pi$)로 한다.(증명생략)

- 푸리에 급수는 단순한 주기함수에 특정한 계수($A_k$)을 곱한것을 모두 합하여 복잡한 주기합수를 나타낸다. 결국 이는 위에서 본 벡터의 개념과 유사하게 적용할 수 있다. 정수배(n)으로 정의되는 다양한 주파수를 가진 단순한 주기함수(축) 간의 직교성 (orthogonality)를 증명하면 그들을 조합하여 모든 주기함수를 나타낼 수 있다. 
다른점은 내적할때 함수는 연속적이기 때문에 함수 곱을 적분을 해주어야 한다(위에 설명) 또한 함수에서는 서로 직교할 수 있는 함수가 무한개가 존재하며, 직교하는 함수 집합을 Orthogonal Set이라고 한다.

구간 [a,b]에서 $f_n(t)$= Orthogonal Set이면, 임의의 함수 $f(t)= \sum_{k=-\infty}^\infty A_k f_n(t)$로 표현 가능하다.

이를 푸리에 급수식에 적용하면 [0,T]구간에서 서로다른 정수배 p,q에 대해 
$ \exp ^{i\cdot \frac{2\pi n}{T}t} $가 직교하는가 증명하면 된다. (생략)

$y(t)=\sum_{k=-\infty}^\infty A_k \, \exp \left( i\cdot 2\pi\frac{k}{T} t \right)$

어떠한 주기함수를 우리는 cos과 sin함수로 표현하였다(cos, sin 함수가 사실상 입력신호에 대해서 기저가 되어주는 함수). 여기서 이 함수들이 직교하는 함수(orthogonal)라는 점을 알아야 한다.
$$
\{ \exp \left(i\cdot 2\pi\frac{k}{T} t\right) \} = orthogonal
$$

벡터의 직교는 해당 벡터를 통해 평면의 모든 좌표를 표현할수 있었다. 함수의 내적은 적분으로 표현할 수 있는데, 만약 구간 [a,b]에서 직교하는 함수는 구간 [a,b]의 모든 함수를 표현할수 있는 것이다.

---
### 푸리에 변환을 하면 time domain이 사라져 해석에 어려움이 생긴다.
따라서 STFT라는 개념이 나온다. STFT는 자른 frame마다 푸리에 변환을 취해, 각각을 시간 순으로 옆으로 쌓아 time domain을 살린다.
이 과정에서 sample by sample로 Window function을 적용한다.

  
Window Function: $$x_w(k) = x(k) \,·\, w(k)$$  
- $x_w(k)$: window signal  
- $x(n)$: original signal 
- $w(n)$: window function  

## DFT vs STFT 

$$ DFT: \hat x(k/N) =\sum_{n=0}^{N-1} x(n) \,·\, e^{-i2πn\frac{k}{N}} $$  

$$ STFT: S(m,k) = \sum_{n=0}^{N-1} x(n+mH) \,·\, w(n)\,·\,e^{-i2πn\frac{k}{N}} $$   

- $k: frequency$    
- $m: time = Frame\,number $   
- $H: hop size $    
   DFT식에서N: all the samples인 반면 STFT의 N은 frame size이다.

## Out put
DFT
- Spectral vector(#frequency bins)
- N complex Foureir coefficients

STFT
- Spectral matrix(#frequency bins, # frames)
- Complex Fourier coefficents
- $frequency bins$(주파수 해상도) = $\frac {frame\,size}{2} +1$(반으로 대칭이여서) 

- $frames =\frac{samples-frame\,size}{hop\,size}+1$

ex)  
Signal = 10K samples  
Frame size = 1000  
Hop size = 500 이라고 할때

frequency bins = 1000 /2 +1 501 -> (0, sampling rate/2) 

frames = (10000 - 1000) / 500 + 1 = 19 

따라서 STFT ->(501, 19) #frequency, number of frame

*frmae size가 작아질수록 해상도는 떨어지고, 시간 속도는 빨라진다

## Visualising sound (Spectrogram)
$ Y(m,k) = |S(m,k)|^2 $   (복소수가 아닌 실수로 표현 하여 스펙트로그램에 보여줌)  

<img src="/images/speech/visual_spectrogram.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 

STFT를 적용하여 구한 spectogram을 시각화 해봤다. x축은 시간, y축은 주파수, 그리고 주파수의 정도를 색깔로 확인할 수 있다. 그런데 값이 너무 미세해서 차이를 파악하고 관찰하기 적합하지 않다.  
그래서 보통 푸리에변환 이후 dB(데시벨) scaling을 적용한 Log-spectogram을 구한다. 다분히 시각적인 이유뿐만 아니라, 사람의 청각 또한 소리를 dB scale 로 인식하기 때문에, 이를 반영하여 spectogram을 나타내는 것이 분석에 용이하다.

<img src="/images/speech/visual_log_spectrogram.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 

(Spectrum의 Y축 magnitude를 제곱해 준것이 power이고,이를 Power spectrum이라고 부른다. 일반적으로는, magnitude에 log scale을 한 데시벨(db) 단위를 많이 사용하고, 이를 log-spectrum이라고 한다. log-spectrum을 세로로 세워서 frame마다 차곡차곡 쌓으면, 푸리에 변환으로 사라졌던 time domain을 복원할 수 있고, 이를 Spectrogram이라고 한다) 

---

# Cepstrum Analysis 
``` 
연관성
spectrum -> Cepstrum 
Quefrency -> Frequency
Liftering- > Filtering
Rhamonic -> Harmonic
``` 
### Cepstrum 
spectrum과 용어가 비슷한데, spectrum의 spectrum 이다. 

Spectrum은 각 주파수 대역의 진폭과 위상을 설명하지만 Cepstrum은 주파수 대역 간의 변동을 특성화해준다. 즉 cepstrum이 spectrum보다 좀 더 깊은 core한 정보를 가지고 있다고 볼 수 있고, spectrum 단이 아닌 cepstrum 단에서 feature extraction을 할 수 있다.

### Define of Cepstrum
$C(x(t)) = F^{-1} [log(F[x(t)])]$ 

- $x(t)$: Time-domain signal  
- F: DFT so $log(F[x(t)])$: log Spectrum
- $F^{-1}$ IFT 

Cepstrum는 푸리에 변환에 의해 신호를 주파수 영역으로 변환한 다음, 다시 이 spectrum을 하나의 신호인 것처럼 다른 변환을 수행하는 것이다. 

이렇게 cepstrum을 구하는 방법은 3가지가 있다고 한다.  
(1) Fourier transform  ->  Complex log  ->  Fourier transform   
(2) Fourier transform  ->  Complex log  ->  Inverse Fourier transform  
(3) Fourier transform  ->  제곱 (power spectrum)  ->  Mel-filter bank  ->  Real log  ->  Discrete cosine transform(DCT)

## Computing the cepstrum

(2)방식  
wave $\rightarrow$(DFT)$\rightarrow$ power spectrum $\rightarrow$(log) $\rightarrow$ log power spectrum $\rightarrow$(IDFT) Cepstrum(x: Quefrency, y:Absolute Magnitude)
* Cepstrum: 고조파를 보여줌(rhamonic), pitch 감지

### Formalsing speech
1. $x(t) = e(t) · h(t)$
- $x(t)$: speech signal
- $e(t)$: 성문펄스(timedomain)
- $h(t)$: 성대주파수

2. $X(t) = E(t) · H(t)$: 스펙트럼 = 성문펄스 스펙트럼 · 성대주파수응답스펙트럼

3. $log(X(t)) = log(E(t)\,·\,H(t)) = log(X(t)) = log(E(t)) + log(H(t))$  

$LogSpectrum(Speech) - Spectral envelope(Vocal tract frequency response) = Spectral detail(Glottal pulse)$   
 즉 Speech = Convolution of vocal tract frequency response with glottal pulse

하지만 성문펄스 스펙트럼(피치에는) 관심이 없고, IDFT후 log(H(t))에서 High value(Hz)를 뽑아낸다.

### Spectral envelope
사람이 음성을 만들 때 사용하는 기관들의 동작과 이들을 각각 공학적으로 모델링한 특징 정보들의 관계를 나타낸다. (과정 생략)
처음 폐에서 만드는 압축된 공기는 백색소음에 가까운 비주기성(aperiodicity) 신호로, 정규분포와 같이 쉽게 사용할 수 있는 확률분포로 모델링할 수 있습니다.  
성대를 통과한 직후의 여기 신호는 유성음/무성음 여부에 따라 구분되며, 유성음의 경우 기본 주파수 등의 특징을 담고 있습니다.  
이후 목, 코, 입, 혀 등의 성도(vocal tract)를 통과하며 발음이 결정되는데, 발음마다 성도의 구조가 달라져 증폭되는 주파수 대역과 감쇠되는 대역 역시 달라지게 됩니다. 이를 스펙트럼 포락선(spectral envelope)이라고 하며, 발음의 종류를 결정하는 주요한 특징으로 꼽힌다.

Log-spectrum에 Spectral envelop한 곳에서 음색에 대한 정보를 가져온다. 

### Formants(Carry identity of sound) 
Formant란 어떠한 소리가 발생했을 때 소리를 구성하는 주파수가 갖는 세기(강도)의 분포를 말합니다. 이 Formant 값을 변경하면 보컬이 노래한 본래의 발음은 변하지 않는 상태에서 소리의 느낌만 바꿀 수 있다. 




## Computing Mel-Frequency Cepstral Coefficients
(3)방식  
wave ->(DFT)-> spectrum ->(log) ->log-Amplitude-spectrum
 ->(Mel-Scaling)->(Discrete Cosine Transform) -> MFCCs

spectrum보다 음성 신호를 더 잘 설명하기에 cepstrum-level에서 feature를 구할 수 있다. spectrum을 만들었다면 이 spectrum에 Mel filter bank라는 필터를 통과 시킨다. Mel-filter의 기본 아이디어는 사람의 청력은 1000Hz 이상의 frequency에 대해서는 덜 민감하므로 1000Hz까지는 Linear하게 그 이상은 log scale로 변환해주는 것이다. 


## DCT (Discrete Cosine Transform)

DCT는 특정 함수를 cosine 함수의 합으로 표현하는 변환이다. 변환하는 이유는, 시간축 상의 화소 data를 주파수축으로 변환 함으로서, 에너지를 한곳으로
집중시킨 영역에서 어떤 처리를 하기 위함이다. 이때 앞쪽(low) cosine 함수의 계수가 변환 전 데이터의 대부분의 정보를 가지고 있고 뒤쪽으로 갈수록 0에 근사해 데이터 압축의 효과를 보인다. 즉 낮은 주파수 쪽으로 에너지 집중현상이 일어난다.
비록 손실이 발생하는 압축이지만, 압축율은 수 10분의 1에서 수 100분의 1까지 가능하다고 한다. 또한 DCT는 inverse 역변환 역시 가능하다.

### Why Discrete Cosine Transform?
- Simplified version of Fourier Transform
- Get real-value coefficent (DFT=complex number)
- Decorrelate energy in different  mel bands
- Reduce number of dimensions to represent spectrum

### How many coefficients?
- Traditionally: first  12-13 coefficients
- First coefficients keep most information(e.g., formants, spectral envelope)
- Use delta and delta-delta MFCCs
- Total 39 coefficeints per frame

---

## Scale
Spectrogram은 Frequency Scale에 대해서 Scaling이 이루어진다. 주파수 영역에 Scaling을 하는 이유는, 인간의 주파수를 인식하는 방식과 연관이 있다. 
일반적으로 사람은 인지기관이 categorical한 구분을 하기 때문 인접한 주파수를 크게 구별하지 못한다. 그래서 우리는 주파수들의 Bin의 그룹을 만들고 이들을 합하는 방식으로 주파수 영역에서 얼마만큼의 에너지가 있는지를 찾아낸다.  
일반적으로는 인간이 적은 주파수에 더 풍부한 정보를 사용하기 때문에, 주파수가 올라갈수록 필터의 폭이 높아지면서 고주파는 거의 고려를 안하게 되고,frequency scale은 어떤 방식을 통해 저주파수대 영역을 고려할 것이가에 생각해야한다.

### Linear frequency scale
일반적으로 single tone(순음)들의 배음 구조를 파악하기 좋다. 하지만 분포가 저주파수 영역에 기울어져(skewed) 있다.

### Mel Scale
사람의 청각기관이 High frequency 보다 low frequency 대역에서 더 민감한  특성을 반영하여 scale변환 함수이다.  
Mel Spectrum은 주파수 단위를 다음 공식에 따라 멜 단위로 바꾼 것을 의미한다.
$$
Mel(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right)
$$
```
일반적으로는 mel-scaled bin을 FFT size보다 조금더 작게 만드는게 일반적이다. 
```

### Mel-filter bank
<img src="/images/speech/mel_triangular.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 
<img src="/images/speech/mel_filter.jpg" width="90%" height="90%" title="waveform" alt="img"/> 

Mel-scale에 linear하게 구간을 N개로 나누어 구현한 triangular filter(window)를 말한다. 1000Hz까지는 Linear하게 변환하다가 그 이후로는 Mel scale triangular filter를 만들어 곱해주어, 지수적으로 넓어지는 것을 볼 수 있다.

보통 26개 혹은 40개 정도의 filter bank를 사용한다. 각 Filter Bank 영역대 마다 Energy값(spectrum power값 평균)을 모두 합하고 log를 취해준다. 이렇게 주파수 영역뿐만 아니라 amplitude 영역에서도 log scaling을 해주는 이유는 사람이 주파수 영역뿐만 아니라 amplitude 영역에서도 log scale로 반응하기 때문이다. 이렇게 하면 결과적으로 filter bank 개수만큼의 Mel scale bin 별로 log power 값들이 구해진다.

## Mel-Spectrogram
Mel fliter bank와 stft의 결과를 곱해주고 dB로 magnitude를 바꿔주었을때의 결과이다.

전체 흐름을 보면
```
1. Mel filter bank를 통과한 값이 n개 이면 n개의 filter bank를 사용했음을 알 수 있고,
2. n개의 Mel scale bin 별로 해당 구간의 power 평균값이 들어가 있음을 알 수 있다. 
3. 이후 log를 취해 amplitude 영역에서의 log scaling을 진행해주고, 마지막으로 DCT 과정을 통해 최종 MFCC를 구할 수 있다. 
```
## MFCC (Mel Frequency Cepstral Coefficient)
Mel-spectrogram을 DCT(Discrete Cosine Transform) 처리하면 얻게되는 coefficient로, mel scale로 변환한 스펙트로그램을 더 적은 값들로 압축하는 과정이라고 볼 수 있다.

#### MFCCs advantages
- Describe the "large" structures of the spectrum
- Ignore fine spectral structures
- Work well in speechand music processing

#### MFCCs disadvantages
- Not robust to noise
- Extensive knowledge engineering
- Not efficient for synthesis

# Process summary
1. 입력 음성을 짧은 frame으로 나눈다.
2. 프레임 각각에 Fourier Transform을 실시해 해당 frame에 담긴 frequency 정보를 추출한다 → spectrum
3. 스펙트럼에 사람의 말소리 인식에 민감한 주파수 영역대는 세밀하게 보고 나머지 영역대는 상대적으로 덜 상세히 분석하는 Mel Filter Bank를 적용한다 → Mel Spectrum
4. 멜 스펙트럼에 로그를 취하면 → log-Mel Spectrum  
- 멜 스펙트럼 혹은 로그 멜 스펙트럼은 태생적으로 feature 내 변수 간 상관관계(correlation)가 존재한다.
- 멜 스케일 필터는 Hz 기준 특정 주파수 영역대의 에너지 정보가 멜 스펙트럼 혹은 로그 멜 스펙트럼의 여러 차원에 영향을 준다.

--- 


# MFCC이 외 다양한 특징
## Band Energy Ratio (BER) 

## Spectral Centroid

## Bandwidth

<img src="/images/speech/ber1.jpg" width="90%" height="90%" title="AE+RMSE" alt="img"/> 












```
REFERENCE
https://ahnjg.tistory.com/47?category=1109653
[DMQA Open Seminar] 

Analysis of Sound dataSound-AI Valerio Velardo |

Analysis of Sound dataSound-AI Valerio Velardo | SKTPLANET

https://hyunlee103.tistory.com/47?category=912459

https://hyunlee103.tistory.com/54

https://hyongdoc.tistory.com/403

https://sanghyu.tistory.com/37?category=1120070

reference: https://tech.kakaoenterprise.com/66

```

