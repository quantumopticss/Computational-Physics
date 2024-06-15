import numpy as np

# 生成一个信号
Fs = 1000  # 采样频率
T = 1 / Fs  # 采样间隔
t = np.arange(0, 10, T)  # 时间向量
f = 50  # 信号频率
x = np.exp(1j*2 * np.pi * f * t)  # 生成正弦波信号

# 进行FFT
X = np.fft.fft(x)

# 获取频率数组
freqs = np.fft.fftfreq(len(X), T)

# 仅保留正频率部分
positive_freqs = freqs[:len(freqs)//2]
positive_X = X[:len(X)//2]

# 可视化结果
import matplotlib.pyplot as plt

plt.plot(positive_freqs, np.abs(positive_X))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()
