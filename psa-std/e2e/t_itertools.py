import itertools
def add(a, b):
    return a + b
print(list(itertools.accumulate([1, 2, 3])))
print(list(itertools.accumulate([1, 2, 3], add)))
print(list(itertools.takewhile(lambda x: x < 3, [1, 2, 3, 4])))
