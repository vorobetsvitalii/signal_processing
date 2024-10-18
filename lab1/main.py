import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


# 1. Функція для обчислення значення f(x)
def f(x, n=2):
    """Обчислення значення функції f(x) = x^n * exp(-x^2 / n)"""
    return x ** n * np.exp(-x ** 2 / n)


# 2. Підпрограми для обчислення коефіцієнтів ряду Фур'є
def a_k(k, n=2):
    """Обчислення коефіцієнту a_k ряду Фур'є."""
    integrand = lambda x: f(x, n) * np.cos(k * x)
    return (1 / np.pi) * quad(integrand, -np.pi, np.pi)[0]


def b_k(k, n=2):
    """Обчислення коефіцієнту b_k ряду Фур'є."""
    integrand = lambda x: f(x, n) * np.sin(k * x)
    return (1 / np.pi) * quad(integrand, -np.pi, np.pi)[0]


# 3. Функція для обчислення наближення ряду Фур'є
def fourier_series(x, N, n=2):
    """Обчислення наближення функції рядом Фур'є з точністю до порядку N."""
    a0 = a_k(0, n) / 2  # Спеціальний випадок для a_0
    series_sum = a0
    for k in range(1, N + 1):
        series_sum += a_k(k, n) * np.cos(k * x) + b_k(k, n) * np.sin(k * x)
    return series_sum


# 4. Головна програма для побудови графіків для різних N
def plot_multiple_approximations(max_N):
    """Побудова графіків початкової функції та наближень ряду Фур'є для кількох значень N."""
    x_values = np.linspace(-np.pi, np.pi, 1000)

    # Графік початкової функції
    plt.plot(x_values, f(x_values), label="Початкова функція", linewidth=2)

    # Побудова графіків наближень для кожного N
    for N in range(1, max_N + 1):
        f_approx = lambda x: fourier_series(x, N)
        plt.plot(x_values, f_approx(x_values), label=f"N = {N}")

    plt.legend()
    plt.title("Наближення ряду Фур'є для різних N")
    plt.grid(True)
    plt.show()


# Головна функція
def main():
    max_N = 10  # Максимальне значення N, до якого будемо обчислювати
    plot_multiple_approximations(max_N)


if __name__ == "__main__":
    main()
