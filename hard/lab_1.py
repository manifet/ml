import numpy as np

# Считаем корень из средней квадратичной ошибки
def get_error(true_val, pred_val):
    return np.sqrt(np.mean((true_val - pred_val) ** 2))

# Стандартный ReLU: отсекаем всё, что меньше нуля
def relu(x):
    return np.maximum(0, x)

# Входные данные (рост, вес или другие признаки)
inp = np.array([
    [15, 10], 
    [15, 15], 
    [15, 20], 
    [25, 10]
])

# Правильные ответы (целевые значения)
true_prediction = np.array([[10, 20, 15, 20]]).T

# Архитектура сети: 2 входа - 10 (скрытый 1) - 7 (скрытый 2) - 1 выход
layer_in_size  = inp.shape[1]
layer_hid1_size = 10
layer_hid2_size = 7
layer_out_size = true_prediction.shape[1]

# Инициализируем веса рандомом от -1 до 1
# Веса между входом и 1-м скрытым слоем
weights_hid1 = 2 * np.random.random((layer_in_size, layer_hid1_size)) - 1

# Веса между 1-м и 2-м скрытыми слоями
weights_hid2 = 2 * np.random.random((layer_hid1_size, layer_hid2_size)) - 1

# Веса между 2-м скрытым слоем и выходом
weights_out = 2 * np.random.random((layer_hid2_size, layer_out_size)) - 1

# Прямой проход

# Слой 1: матричное умножение + активация
layer1_act = relu(np.dot(inp, weights_hid1))

# Слой 2: берем выход первого и снова через ReLU
layer2_act = relu(np.dot(layer1_act, weights_hid2))

# Выходной слой: здесь обычно активация не нужна, если решаем задачу регрессии
prediction = np.dot(layer2_act, weights_out)

# Смотрим че получилось
print("Активации 1-го слоя:\n", layer1_act)
print("\nАктивации 2-го слоя:\n", layer2_act)
print("\nПредсказание системы:\n", prediction)

# Считаем итоговую ошибку 
error = get_error(true_prediction, prediction)
print(f"\nОшибка (RMSE): {error:.4f}")