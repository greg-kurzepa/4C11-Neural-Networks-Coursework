import itertools

d = {
    "h1": {'a': [0, 1, 2], 'b': [0, 1]},
    "h2" : {'1': ['h', 'e'], '2': ['l', 'o']}
}

def product_dict2(**kwargs):
    """Finds the Cartesian product of the given 2-nested dictionary"""
    keys = kwargs.keys()
    for instance in itertools.product(*[product_dict(**value) for value in kwargs.values()]):
        yield dict(zip(keys, instance))

def product_dict(**kwargs):
    """Finds the Cartesian product of the given dictionary.
    see https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists"""
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

print(list(product_dict2(**d)))