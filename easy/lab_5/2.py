import numpy as np

# Простая нейронка (линейная регрессия)
def neural_networks(inp, weights):
    return inp.dot(weights)

def get_error(target, pred):
    return (target - pred) ** 2

def train(inp, targets, weights, lr, epochs):
    # Нормализуем данные, чтобы LR не ломал веса при больших значениях
    # Делим на 200 (макс. рост/вес), чтобы значения были от 0 до 1
    inp_norm = inp / 200
    
    for i in range(epochs):
        epoch_error = 0
        delta = np.zeros_like(weights)
        
        for j in range(len(inp_norm)):
            current_x = inp_norm[j]
            target = targets[j]
            
            prediction = neural_networks(current_x, weights)
            epoch_error += get_error(target, prediction)
            
            # Считаем градиент для текущего шага
            delta += (prediction - target) * current_x
            
        # Обновляем веса (среднее по всей выборке)
        weights -= (delta / len(inp_norm)) * lr
        
        # Печатаем лог раз в 100 эпох, чтобы не спамить в консоль
        if i % 100 == 0:
            print(f"Эпоха {i}, Ошибка: {epoch_error:.5f}")
            
    return weights

# Входные данные: [рост, вес]
inp = np.array([
    [150, 40], [140, 35], [155, 45], 
    [185, 95], [145, 40], [195, 100], 
    [180, 95], [170, 80], [160, 90]
])

# 0 - женщина, 100 - мужчина
true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

# Начальные веса
weights = np.array([0.1, 0.1])

# Оптимальный LR после нормализации. 
# Если поставить 1.0 — обучится мгновенно. 
# Если 0.0001 — будет тупить.
learning_rate = 0.1 
epochs = 500

# Запуск обучения
weights = train(inp, true_predictions, weights, learning_rate, epochs)

# Проверка результатов
def test_model(h, w, w_final):
    # Не забываем нормализовать тестовые данные так же, как обучающие
    res = neural_networks(np.array([h, w]) / 200, w_final)
    return f"Рост {h}, Вес {w} -> Прогноз: {res:.2f}"

print("-" * 20)
print(test_woman := test_model(150, 45, weights))
print(test_man := test_model(170, 85, weights))