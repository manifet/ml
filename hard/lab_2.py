import numpy as np

# Активация: всё что < 0 зануляем
def relu(x):
    return np.maximum(0, x)

# Производная ReLU: 1 для положительных, 0 для остальных
def relu_deriv(x):
    return (x > 0).astype(float)

# Исходные данные и таргеты
inp = np.array([[15, 10], [15, 18], [15, 20], [25, 10]])
true_prediction = np.array([[15, 18, 20, 25]]).T  # Сразу делаем вектор-столбец

# Параметры слоев
in_size = inp.shape[1]
out_size = 1

def train(hidden_size, lr, epochs):
    np.random.seed(100)
    
    # Инициализация весов
    w_hid = 2 * np.random.random((in_size, hidden_size)) - 1
    w_out = 2 * np.random.random((hidden_size, out_size)) - 1
    
    for epoch in range(epochs):
        epoch_error = 0
        
        for i in range(len(inp)):
            layer_in = inp[i:i+1]
            y_true = true_prediction[i:i+1]
            
            # Forward pass
            layer_hid = relu(np.dot(layer_in, w_hid))
            layer_out = np.dot(layer_hid, w_out)
            
            # Считаем MSE
            error = np.sum((layer_out - y_true) ** 2)
            epoch_error += error
            
            # Дельта выхода (просто разница)
            delta_out = layer_out - y_true
            
            # Дельта скрытого слоя: тянем ошибку назад через веса и бьем по ReLU
            delta_hid = delta_out.dot(w_out.T) * relu_deriv(layer_hid)
            
            # Градиентный спуск: обновляем веса
            w_out -= lr * layer_hid.T.dot(delta_out)
            w_hid -= lr * layer_in.T.dot(delta_hid)
            
    return epoch_error

print("1: Размеры скрытого слоя")
for size in [2, 3, 5, 8, 12]:
    err = train(hidden_size=size, lr=0.0001, epochs=300)
    print(f"Скрытых нейронов: {size:2d} | Итоговая ошибка: {err:.6f}")

print("\nСкорость обучения")
for rate in [1e-5, 1e-4, 1e-3, 1e-2]:
    err = train(hidden_size=5, lr=rate, epochs=200)
    print(f"LR: {rate:.5f} | Итоговая ошибка: {err:.6f}")

print("\n3: Количество эпох")
for eps in [50, 100, 300, 1000]:
    err = train(hidden_size=5, lr=0.0001, epochs=eps)
    print(f"Эпох: {eps:4d} | Итоговая ошибка: {err:.6f}")