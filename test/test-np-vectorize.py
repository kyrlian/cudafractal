import numpy as np

## Basic vectorized

def basic_func(x, y):
    # print(f"basic_func: x:{x}, y:{y}")
    return x + y

vectorized_basic_func = np.vectorize(basic_func, otypes=[np.int32])

x_array = [1, 2, 3, 4]
y_array = [10]#, 20, 30, 40]
result_array = vectorized_basic_func(x_array,y_array)
print(f"basic_func result_array:{result_array}")

##### in place modif

def array_func(x, y, v):
    return x+y, v+1

vectorized_array_func = np.vectorize(array_func, otypes=[np.int32,np.int32])

x_array = [1, 2, 3, 4]
y_array = [10, 20, 30, 40]
v_array = 2
result_array2 = vectorized_array_func(x_array,y_array,v_array)
print(f"vectorized_array_func result_array:{result_array2}")
