import numpy as np
import matplotlib.pyplot as plt
import time


# Частина І

# Функція для обчислення одного (k-го) члена ряду Фур'є у тригонометричній формі з підрахунком операцій
def compute_fourier_term(f, k, N, operation_count):
    Ak = 0
    Bk = 0
    for n in range(N):
        Ak += f[n] * np.cos(2 * np.pi * k * n / N)
        Bk += f[n] * np.sin(2 * np.pi * k * n / N)

        # Оновлюємо кількість операцій
        operation_count['multiplications'] += 2  # 2 множення: для Ak і Bk
        operation_count['additions'] += 2  # 2 додавання: для Ak і Bk
    return Ak, Bk


# Функція для обчислення коефіцієнта Фур'є Ck з підрахунком операцій
def compute_fourier_coefficient(f, k, N, operation_count):
    Ak, Bk = compute_fourier_term(f, k, N, operation_count)
    Ck = Ak + 1j * Bk
    # Оновлюємо кількість операцій для складання комплексного числа
    operation_count['additions'] += 1  # Додавання комплексної частини
    return Ck


# Генерація випадкового вхідного сигналу
n = 2
N = 10 + n
f = np.random.random(N)

# Ініціалізація лічильника операцій
operation_count = {'multiplications': 0, 'additions': 0}

# Обчислення ДПФ з підрахунком операцій
start_time = time.time()
Ck = [compute_fourier_coefficient(f, k, N, operation_count) for k in range(N)]
end_time = time.time()

# Час обчислення
print(f"Час обчислення: {end_time - start_time} секунд")
print(f"Кількість операцій множення: {operation_count['multiplications']}")
print(f"Кількість операцій додавання: {operation_count['additions']}")

# Спектр амплітуд і фаз
amplitudes = np.abs(Ck)
phases = np.angle(Ck)

# Побудова графіка спектру амплітуд
plt.subplot(2, 1, 1)
plt.stem(range(N), amplitudes)
plt.title('Спектр амплітуд')
plt.xlabel('k')
plt.ylabel('|Ck|')

# Побудова графіка фазового спектру
plt.subplot(2, 1, 2)
plt.stem(range(N), phases)
plt.title('Фазовий спектр')
plt.xlabel('k')
plt.ylabel('arg(Ck)')

plt.tight_layout()
plt.show()

# Частина ІІ

# Генерація N рівновіддалених відліків у 8-бітній двійковій системі
N_part2 = 96 + n
binary_signal = np.random.choice([0, 1], size=N_part2)

# Обчислення ДПФ для відліків
Ck_part2 = np.fft.fft(binary_signal)
amplitudes_part2 = np.abs(Ck_part2)
phases_part2 = np.angle(Ck_part2)

# Побудова графіка відтвореного аналогового сигналу
t = np.linspace(0, 1, N_part2, endpoint=False)
reconstructed_signal = np.fft.ifft(Ck_part2).real

# Відображення графіка відтвореного сигналу
plt.plot(t, reconstructed_signal)
plt.title('Часова залежність відтвореного сигналу')
plt.xlabel('Час')
plt.ylabel('Амплітуда')
plt.show()

# Виведення результатів
print(f"Модулі Cn: {amplitudes_part2}")
print(f"Аргументи Cn: {phases_part2}")
