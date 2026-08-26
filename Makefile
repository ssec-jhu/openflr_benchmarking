open_paren := \(

GPU_NAME := $(shell nvidia-smi -L | sed -n '1p' | sed -E 's/^GPU [0-9]+: (.*) $(open_paren)UUID.*/\1/' | tr ' ' '_')

print-gpu:
	@echo "GPU 0 name: $(GPU_NAME)"

all: time-jax-v1 time-jax-v2 time-torch-v1 time-torch-v2 time-numpy-v1 time-numpy-v2 time-mojo-v1 time-mojo-v2

time-jax-v1:
	export CUDA_VISIBLE_DEVICES=0; \
	uv run main.py --use_openflr --backend jax --n_iters 20 > $(GPU_NAME)_jax_v1.txt

time-jax-v2:
	export CUDA_VISIBLE_DEVICES=0; \
	uv run main.py --use_openflr_v2 --backend jax --n_iters 20 > $(GPU_NAME)_jax_v2.txt

time-torch-v1:
	export CUDA_VISIBLE_DEVICES=0; \
	uv run main.py --use_openflr --backend torch --n_iters 20 > $(GPU_NAME)_torch_v1.txt

time-torch-v2:
	export CUDA_VISIBLE_DEVICES=0; \
	uv run main.py --use_openflr_v2 --backend torch --n_iters 20 > $(GPU_NAME)_torch_v2.txt

time-numpy-v1:
	export CUDA_VISIBLE_DEVICES=0; \
	uv run main.py --use_openflr --backend numpy --n_iters 20 > $(GPU_NAME)_numpy_v1.txt

time-numpy-v2:
	export CUDA_VISIBLE_DEVICES=0; \
	uv run main.py --use_openflr_v2 --backend numpy --n_iters 20 > $(GPU_NAME)_numpy_v2.txt

time-mojo-v1:
	cd openflr-mojo && pixi run mojo run src/main.mojo -- v1 20 > ../$(GPU_NAME)_mojo_v1.txt

time-mojo-v2:
	cd openflr-mojo && pixi run mojo run src/main.mojo -- v2 20 > ../$(GPU_NAME)_mojo_v2.txt
