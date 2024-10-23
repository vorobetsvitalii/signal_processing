import numpy as np
import matplotlib.pyplot as plt
import time


# ----------- ШПФ -----------
def fft_recursive(f, N, operation_count):
    if N <= 1:
        return f
    even_part = fft_recursive(f[0::2], N // 2, operation_count)
    odd_part = fft_recursive(f[1::2], N // 2, operation_count)

    combined = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        exp_factor = np.exp(-2j * np.pi * k / N)
        combined[k] = even_part[k] + exp_factor * odd_part[k]
        combined[k + N // 2] = even_part[k] - exp_factor * odd_part[k]

        operation_count['multiplications'] += 1  # Множення на експоненту
        operation_count['additions'] += 2  # Додавання та віднімання

    return combined


# ----------- Генерація випадкових сигналів -----------
n = 2
N = 10 + n
f = np.random.random(N)  # Генерація випадкового сигналу

# ----------- ШПФ (швидке перетворення) -----------
operation_count_fft = {'multiplications': 0, 'additions': 0}
start_time_fft = time.time()
Ck_fft = fft_recursive(f, N, operation_count_fft)
end_time_fft = time.time()

# ----------- Результати обчислень -----------
print("--------- ШПФ (швидке перетворення Фур'є) ---------")
print(f"Час обчислення ШПФ: {end_time_fft - start_time_fft} секунд")
print(f"Кількість операцій множення для ШПФ: {operation_count_fft['multiplications']}")
print(f"Кількість операцій додавання для ШПФ: {operation_count_fft['additions']}")

# ----------- Побудова графіків -----------

amplitudes_fft = np.abs(Ck_fft)
phases_fft = np.angle(Ck_fft)

# Амплітуди ШПФ
plt.subplot(2, 1, 1)
plt.stem(range(N), amplitudes_fft, linefmt='r-', markerfmt='ro', basefmt=" ")
plt.axhline(0, color='black', linewidth=1)  # Додаємо лінію осі X
plt.title('Спектр амплітуд (ШПФ)')
plt.xlabel('k')
plt.ylabel('|Ck|')

# Фази ШПФ
plt.subplot(2, 1, 2)
plt.stem(range(N), phases_fft, linefmt='r-', markerfmt='ro', basefmt=" ")
plt.axhline(0, color='black', linewidth=1)  # Додаємо лінію осі X
plt.title('Фазовий спектр (ШПФ)')
plt.xlabel('k')
plt.ylabel('arg(Ck)')

plt.tight_layout()
plt.show()
