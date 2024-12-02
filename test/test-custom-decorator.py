def decorator(before:str="Before",after:str="After"):
    def wrapper(func):
        def decorated_func(*args, **kwargs):
            print(f"Pre:{before}")
            func(*args, **kwargs)
            print(f"Post:{after}")
        return decorated_func
    return wrapper

@decorator(before="a",after="b")
def hello2(msg:str):
    print(f"Hello {msg}")

hello2("world")
