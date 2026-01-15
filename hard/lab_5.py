import numpy as np

# Набор функций
def sigmoid(x):      return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

def tanh(x):         return np.tanh(x)
def tanh_deriv(x):    return 1 - x**2

def relu(x):         return np.maximum(0, x)
def relu_deriv(x):    return (x > 0).astype(float)

# Данные XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0],   [1],   [1],   [0]])

def train_xor(act_type="sigmoid", h_size=4, lr=0.1, epochs=10000):
    np.random.seed(42)
    
    # Инициализация весов
    w0 = 2 * np.random.random((2, h_size)) - 1
    w1 = 2 * np.random.random((h_size, 1)) - 1
    
    # Настройка под активацию
    if act_type == "sigmoid":
        f, df = sigmoid, sigmoid_deriv
        y_true = Y
    elif act_type == "tanh":
        f, df = tanh, tanh_deriv
        y_true = Y * 2 - 1 # Конвертируем 0/1 в -1/1
    else: # relu
        f, df = relu, relu_deriv
        y_true = Y

    for i in range(epochs):
        # Forward
        l1 = f(np.dot(X, w0))
        l2 = f(np.dot(l1, w1)) if act_type != "relu" else np.dot(l1, w1)
        
        # Backprop
        l2_error = l2 - y_true
        l2_delta = l2_error * (df(l2) if act_type != "relu" else 1.0)
        
        l1_error = l2_delta.dot(w1.T)
        l1_delta = l1_error * df(l1)
        
        # Обновление
        w1 -= lr * l1.T.dot(l2_delta)
        w0 -= lr * X.T.dot(l1_delta)
        
    # Финальная проверка 
    final_error = np.mean(np.square(l2 - y_true))
    return final_error

# Тестирование
results = []
configs = [
    ("sigmoid", 2), ("sigmoid", 4),
    ("tanh",    2), ("tanh",    4),
    ("relu",    2), ("relu",    4)
]

print(f"{'Активация':<10} | {'Скрытый слой':<12} | {'Ошибка (MSE)':<12}")
print("-" * 40)

for act, size in configs:
    err = train_xor(act, size)
    print(f"{act:<10} | {size:<12} | {err:.6f}")