import numpy as np

# Простая линейная модель
def predict(inp, w):
    return inp.dot(w)

def get_error(target, pred):
    return (target - pred) ** 2

# Данные: [рост, вес]. Делим на 200 для нормализации (чтобы веса не скакали)
X = np.array([
    [150, 40], [140, 35], [155, 45], # Ж
    [185, 95], [145, 40], [195, 100], # М
    [180, 95], [170, 80], [160, 90]   # М
]) / 200

# 0 - женщина, 100 - мужчина
y = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

weights = np.array([0.1, 0.1])
lr = 0.01 # шаг обучения

for epoch in range(500):
    # Считаем предсказания сразу для всей выборки
    preds = X.dot(weights)
    errors = preds - y
    loss = np.mean(errors**2)
    
    # Считаем градиент (среднее по всем примерам)
    gradient = X.T.dot(errors) / len(y)
    weights -= gradient * lr
    
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, Ошибка: {loss:.4f}")

print("-" * 20)
# Тестим на новых данных (не забываем про нормализацию /200)
test_woman = np.array([150, 45]) / 200
test_man = np.array([170, 85]) / 200

print(f"Тест (женщина): {predict(test_woman, weights):.2f}")
print(f"Тест (мужчина): {predict(test_man, weights):.2f}")