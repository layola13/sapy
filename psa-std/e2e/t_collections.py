import collections
print(dict(collections.Counter(["a", "b", "b"])))
c = collections.Counter(["a", "b", "b"])
print(c["b"])
print(sum(c.values()))
