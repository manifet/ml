import numpy as np

# Обычная линейная модель
def neural_networks(inp, weights):
    return inp.dot(weights)

def get_error(target, pred):
    return (target - pred) ** 2

def train_model(inp, targets, weights, lr, epochs):
    for i in range(epochs):
        error = 0
        delta = np.zeros_like(weights)
        
        for j in range(len(inp)):
            prediction = neural_networks(inp[j], weights)
            error += get_error(targets[j], prediction)
            
            # Накапливаем коррекцию весов
            delta += (prediction - targets[j]) * inp[j]
        
        # Обновляем веса (усредняем по батчу)
        weights -= (delta / len(inp)) * lr
        
        # Логируем только каждую сотую эпоху, чтобы не тормозить процесс
        if i % 100 == 0:
            print(f"Эпоха: {i} | Ошибка: {error:.4f}")
            
        # Защита от взрыва градиента (если ошибка улетела в бесконечность)
        if np.isinf(error) or np.isnan(error):
            print("Обучение прервано: веса разошлись!")
            break
            
    return weights

# Данные: [рост, вес]
X = np.array([
    [150, 40], [140, 35], [155, 45], 
    [185, 95], [145, 40], [195, 100], 
    [180, 95], [170, 80], [160, 90]
])

y = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])

# Начальные параметры
w = np.array([0.2, 0.3])
lr = 0.00001
num_epochs = 1000 # Оптимально. 100 - мало, 100000 - риск переобучения или вылета

# Запуск
w = train_model(X, y, w, lr, num_epochs)

# Проверка
def predict_gender(h, w_val, trained_w):
    res = neural_networks(np.array([h, w_val]), trained_w)
    return f"Результат для {h}см/{w_val}кг: {res:.2f}"

print("-" * 30)
print(predict_gender(150, 45, w)) # Ж
print(predict_gender(170, 85, w)) # М