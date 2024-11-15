# cudafractal

## Requirements

- The cuda version requires [cuda](https://developer.nvidia.com/cuda-downloads) - you can install only the compiler (CUDA/Development/Compiler) & libraries (CUDA/Runtime/Libraries)

## Install as uv tool from git

```sh
uv tool install git+https://github.com/kyrlian/fractal-python-cuda.git
```

Run with 

```sh
fractal
```

## Or clone and run with uv

```sh
git clone https://github.com/kyrlian/fractal-python-cuda.git
cd fractal-python-cuda
```
Run with 
```sh
uv run ui/main_ui.py
```

## Or with pip

```sh
pip install -r requirements.txt
```
Run with 
```sh
python ui/main_ui.py
```