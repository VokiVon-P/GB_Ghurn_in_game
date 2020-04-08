import time
from datetime import timedelta


def time_format(sec):
    return str(timedelta(seconds=sec))


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        return_value = func(*args, **kwargs)
        end = time.time()

        print(f'\nВремя выполнения {func.__name__} : {(end - start):0.2f} секунды\n')
        return return_value

    return wrapper
