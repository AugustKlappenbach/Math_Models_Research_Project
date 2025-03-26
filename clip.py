from numba import njit

@njit(cache=True)
def clip_num(num, a, b):
    if a < num < b:
        return num
    elif num < a:
        return a
    else:
        return b