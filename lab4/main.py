import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Константи
n = 2
N = 100 * n

# Функція f(t) для n = 2
def f(t):
    return t ** 4

# Функція для обчислення F(w_k)
def F_wk(k, T):
    w_k = 2 * np.pi * k / T

    # Дійсна частина
    Re_F, _ = quad(lambda t: f(t) * np.cos(-w_k * t), -N, N)

    # Уявна частина
    Im_F, _ = quad(lambda t: f(t) * np.sin(-w_k * t), -N, N)

    return Re_F, Im_F

# Функція для обчислення амплітуди |F(w_k)|
def amplitude(Re_F, Im_F):
    return np.sqrt(Re_F ** 2 + Im_F ** 2)

# Значення T, для яких будуть побудовані графіки
T_values = [4, 8, 16, 32, 64, 128]
k_values = range(20)  # кількість гармонік для обчислення

# Побудова графіків
plt.figure(figsize=(14, 6))

# Графік Re(F(w_k))
plt.subplot(1, 2, 1)
for T in T_values:
    Re_F_values = []
    for k in k_values:
        Re_F, Im_F = F_wk(k, T)
        Re_F_values.append(Re_F)
    plt.plot(k_values, Re_F_values, '-o', label=f'T={T}')

plt.xlabel('k')
plt.ylabel('Re(F(w_k))')
plt.title('Re(F(w_k)) для різних значень T')
plt.legend()

# Графік |F(w_k)|
plt.subplot(1, 2, 2)
for T in T_values:
    amplitude_values = []
    for k in k_values:
        Re_F, Im_F = F_wk(k, T)
        amplitude_values.append(amplitude(Re_F, Im_F))
    plt.plot(k_values, amplitude_values, '-o', label=f'T={T}')

plt.xlabel('k')
plt.ylabel('|F(w_k)|')
plt.title('|F(w_k)| для різних значень T')
plt.legend()

plt.tight_layout()
plt.show()
