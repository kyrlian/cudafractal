### Inspect decorator

print("\n### inspect_decorator\n")

def dummy_func():
    pass

def inspect_decorator(*args, **kwargs):
    # this is a no-op decorator to inspect what the decorator receives
    print(" inspect_decorator")
    print(f"  *args={args}")
    print(f"  **kwargs={kwargs}")
    if len(args) == 1 and type(args[0]) is type(dummy_func):
        #if I have and arg and it's a function, return it
        return args[0]
    else:
        # If I don't have the func as arg -> return a wrapper that would take the function and decorate it (here doing nothing, just returning the function)
        def wrapper(func):
            return func
        return wrapper

@inspect_decorator #without args, the decorator DOES NOT receive the function as in it's arguments
def hello_inspect_decorator(msg:str):
    print(f"Hello {msg}")

hello_inspect_decorator("inspect_decorator without argument")

@inspect_decorator("an arg") #with an arg
def hello_inspect_decorator(msg:str):
    print(f"Hello {msg}")

hello_inspect_decorator("inspect_decorator with an argument")

### Real dummy decorator

print("\n### wrapper_decorator\n")

def wrapper_decorator(before:str="Before",after:str="After"):
    # this is a real decorator that expects two arguments, and will use the wrapper to catch the function (as it's not passed as an arg)
    def wrapper(func):
        print(f"wrapper_decorator: wrapping {func}")
        def decorated_func(*args, **kwargs):
            print(f"Pre:{before}")
            func(*args, **kwargs)
            print(f"Post:{after}")
        return decorated_func
    return wrapper

@wrapper_decorator(before="a",after="b")
def hello_wrapper_decorator(msg:str):
    print(f"Hello {msg}")

hello_wrapper_decorator("wrapper_decorator")
