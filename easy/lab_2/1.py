import numpy as np

# 1-2: Одномерный массив и скалярные операции
def f_1_2():
    arr = np.array([2, 3, 4, 9, 1, 0])
    print(f"Оригинал: {arr}")
    # NumPy сам применяет операцию к каждому элементу
    print(f"Умножение на 2: {arr * 2}")

# 3-4: Матрицы и их умножение
def f_3_4():
    N = 3
    # Генерируем случайные матрицы
    m1 = np.random.randint(-100, 100, size=(N, N))
    m2 = np.random.randint(-100, 100, size=(N, N))
    
    print("Матрица 1:\n", m1)
    print("Матрица 2:\n", m2)
    
    print("Матричное произведение (@):\n", m1 @ m2)

# 5: Генерация выборки
def get_random_array(n=10):
    return np.random.randint(-500, 500, size=n)

# 6: Фильтрация через булевы маски
def f_6():
    arr = get_random_array()
    # Выбираем только четные элементы
    even = arr[arr % 2 == 0]
    print(f"Массив: {arr}")
    print(f"Четные: {even}")

# 7: Базовая статистика
def f_7():
    arr = get_random_array()
    print(f"Массив: {arr}")
    print(f"Среднее: {arr.mean():.2f}")
    print(f"Станд. отклонение: {arr.std():.2f}")
    print(f"Max: {arr.max()}, Min: {arr.min()}")

print("Tasks 1 & 2")
f_1_2()
print("\nTasks 3 & 4")
f_3_4()
print("\nTask 6")
f_6()
print("\nTask 7")
f_7()