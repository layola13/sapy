import functools
def add(a, b):
    return a + b
print(functools.reduce(add, [1, 2, 3, 4]))
print(functools.reduce(add, [1, 2, 3, 4], 100))
