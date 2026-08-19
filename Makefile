olaf:
	@echo "Running Olaf reconstruction..."
	export CUDA_VISIBLE_DEVICES=0; \
	uv run --group olaf main.py --use_olaf

openflr-torch:
	@echo "Running OpenFLR reconstruction..."
	export FLFM_TIME_RECONSTRUCTION=1; \
	export FLFM_DEFAULT_RL_ITERS=10; \
	export CUDA_VISIBLE_DEVICES=0; \
	export FLFM_BACKEND=torch; \
	uv run --group openflr main.py --use_openflr

openflr-jax:
	@echo "Running OpenFLR reconstruction..."
	export FLFM_TIME_RECONSTRUCTION=1; \
	export FLFM_DEFAULT_RL_ITERS=10; \
	export CUDA_VISIBLE_DEVICES=0; \
	export FLFM_BACKEND=jax; \
	uv run --group openflr main.py --use_openflr
