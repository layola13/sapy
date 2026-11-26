import random
random.seed(42)
print(random.random() == 0.6394267984578837)
print(random.randint(1, 100))
print(random.getrandbits(32))
print(random.randrange(5, 50, 3))
xs = list(range(10))
random.shuffle(xs)
print(xs)
print(random.sample(xs, 3))
words = ["a", "b", "c"]
print(random.choice(words))
print(random.choices(words))
print(random.choices(words, [1, 2, 3], k=4))
