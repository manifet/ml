import numpy as np

# Стартовые веса
weights = np.array([0.2, 0.3])

def neural_networks(inp, weights):
    return inp.dot(weights)

# Перешли на RMSE для оценки отклонения
def get_error(target, pred):
    # Корень из среднего квадрата ошибки
    return np.sqrt(np.mean((target - pred) ** 2))

def gradient_descent(inp, targets, w, lr, epochs):
    for i in range(epochs):
        epoch_rmse = 0
        delta = np.zeros_like(w)
        
        for j in range(len(inp)):
            current_x = inp[j]
            target = targets[j]
            
            prediction = neural_networks(current_x, w)
            # Считаем RMSE для мониторинга
            epoch_rmse += get_error(target, prediction)
            
            # Градиент все еще считаем по разности (линейно)
            delta += (prediction - target) * current_x
        
        # Обновляем веса
        w -= (delta / len(inp)) * lr
        
        # Выводим статус раз в 5000 эпох, чтобы не вешать консоль
        if i % 5000 == 0:
            print(f"Эпоха {i} | RMSE: {epoch_rmse/len(inp):.5f}")
            
    return w

# Обучающая выборка [рост, вес]
inp = np.array([
    [150, 40], [140, 35], [155, 45], 
    [185, 95], [145, 40], [195, 100], 
    [180, 95], [170, 80], [160, 90]
])

# Метки: 0 - Ж, 100 - М
true_labels = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

lr = 0.00001
epochs = 100000 

# Запуск обучения
weights = gradient_descent(inp, true_labels, weights, lr, epochs)

print("-" * 30)
# Тестируем
def check(h, w_val):
    res = neural_networks(np.array([h, w_val]), weights)
    return f"Вход {h}/{w_val} -> Прогноз: {res:.2f}"

print(check(150, 45))
print(check(170, 85))