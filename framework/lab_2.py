import numpy as np

class Tensor:
    # Счетчик для уникальных ID
    _next_id = 0
    
    def __init__(self, data, creators=None, op=None, autograd=False, id=None):
        # Оборачиваем данные в numpy array, если они еще не там
        self.data = np.array(data)
        self.creators = creators
        self.op = op
        self.autograd = autograd
        self.grad = None
        
        # Присваиваем ID по порядку, если не задан вручную
        if id is None:
            id = Tensor._next_id
            Tensor._next_id += 1
        self.id = id
        
        # Считаем, сколько раз этот тензор используется в других операциях
        self.children = {}
        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def check_grads(self):
        for child_id in self.children:
            if self.children[child_id] != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if not self.autograd:
            return

        # Если мы в самом конце графа, начинаем с единичного градиента
        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        # Уменьшаем счетчик ожидаемых градиентов от потомков
        if grad_origin is not None:
            if self.children[grad_origin.id] > 0:
                self.children[grad_origin.id] -= 1
            else:
                raise Exception(f"Ошибка: тензор {grad_origin.id} не является ребенком {self.id}")

        # Накапливаем градиент
        if self.grad is None:
            self.grad = grad
        else:
            self.grad.data += grad.data

        # Если все дети прошли, идем дальше по графу
        if self.creators is not None and (self.check_grads() or grad_origin is None):
            
            if self.op == "+":
                # В сложении градиент просто дублируется обоим родителям
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)

            elif self.op == "-":
                # Первому плюс, второму минус
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(Tensor(-self.grad.data), self)

            elif self.op == "*":
                # Правило произведения: d(ab) = b*da + a*db
                self.creators[0].backward(Tensor(self.grad.data * self.creators[1].data), self)
                self.creators[1].backward(Tensor(self.grad.data * self.creators[0].data), self)

            elif self.op == "@":
                # Матричное умножение
                grad_0 = self.grad.data @ self.creators[1].data.T
                grad_1 = self.creators[0].data.T @ self.grad.data
                self.creators[0].backward(Tensor(grad_0), self)
                self.creators[1].backward(Tensor(grad_1), self)
            
            elif self.op == "neg":
                # При отрицании градиент меняет знак
                self.creators[0].backward(Tensor(-self.grad.data), self)

    # Магические методы для удобства операций
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", autograd=True)
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", autograd=True)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", autograd=True)
        return Tensor(self.data * other.data)

    def __matmul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data @ other.data, [self, other], "@", autograd=True)
        return Tensor(self.data @ other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(-self.data, [self], "neg", autograd=True)
        return Tensor(-self.data)

    def __repr__(self):
        return f"Tensor(id={self.id}, data={self.data}, grad={self.grad.data if self.grad else None})"


# 1. Проверка правильности работы операции сложения
a = Tensor([1, 2], autograd=True)
b = Tensor([3, 4], autograd=True)
c = a + b
c.backward()
print(f"Сложение: a.grad={a.grad.data}, b.grad={b.grad.data}")

# 2. Проверка правильности работы операции вычитания
a = Tensor([10, 20], autograd=True)
b = Tensor([1, 2], autograd=True)
c = a - b
c.backward()
print(f"Вычитание: a.grad={a.grad.data}, b.grad={b.grad.data}")

# 3. Проверка правильности работы операции умножения
a = Tensor([3, 4], autograd=True)
b = Tensor([5, 6], autograd=True)
c = a * b
c.backward()
print(f"Умножение: a.grad={a.grad.data}, b.grad={b.grad.data}")

# 4. Проверка правильности работы операции отрицания
a = Tensor([5, -10], autograd=True)
b = -a
b.backward()
print(f"Отрицание: a.grad={a.grad.data}")