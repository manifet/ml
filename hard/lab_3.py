import numpy as np
from keras.datasets import mnist

# Активация и её производная
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# 1. Загрузка и подготовка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Гиперпараметры
train_size = 1000
test_size = 10000
input_size = 784  # 28*28 пикселей
hidden_size = 50
output_size = 10  # Цифры 0-9

# Нормализуем данные (0-255 - 0-1) и вытягиваем в вектор
train_images = x_train[:train_size].reshape(train_size, input_size) / 255.0
test_images = x_test[:test_size].reshape(test_size, input_size) / 255.0

# One-Hot Encoding для меток 
train_labels = np.zeros((train_size, output_size))
train_labels[np.arange(train_size), y_train[:train_size]] = 1

test_labels = y_test[:test_size]

# 2. Инициализация весов
np.random.seed(1) 
# Используем небольшой разброс, чтобы не "взрывать" градиенты
w_in_hid = 0.2 * np.random.random((input_size, hidden_size)) - 0.1
w_hid_out = 0.2 * np.random.random((hidden_size, output_size)) - 0.1

lr = 0.005
epochs = 40

# 3. Цикл обучения
for epoch in range(epochs):
    correct_count = 0
    
    for i in range(train_size):
        # Forward pass
        layer_in = train_images[i:i+1]
        layer_hid = relu(np.dot(layer_in, w_in_hid))
        layer_out = np.dot(layer_hid, w_hid_out)

        # Считаем точность на тренировке
        if np.argmax(layer_out) == np.argmax(train_labels[i:i+1]):
            correct_count += 1

        # Backward pass
        # Ошибка выхода
        delta_out = layer_out - train_labels[i:i+1]
        
        # Ошибка скрытого слоя (пробрасываем через веса и ReLU)
        delta_hid = delta_out.dot(w_hid_out.T) * relu_deriv(layer_hid)

        # Обновление весов 
        w_hid_out -= lr * layer_hid.T.dot(delta_out)
        w_in_hid -= lr * layer_in.T.dot(delta_hid)

    if (epoch + 1) % 10 == 0:
        print(f"Эпоха {epoch+1} | Точность: {correct_count/train_size:.2%}")

# 4. Проверка на тестовых данных
test_correct = 0
for i in range(test_size):
    layer_hid = relu(np.dot(test_images[i:i+1], w_in_hid))
    layer_out = np.dot(layer_hid, w_hid_out)
    
    if np.argmax(layer_out) == test_labels[i]:
        test_correct += 1

print(f"\nИтоговая точность на тесте: {test_correct/test_size:.2%}")