BUILD_DIR := build
HWY_PATH  := ./external/highway

RUN_ARGS  := $(filter-out gcc-spmv clang-spmv icx-spmv run clean,$(MAKECMDGOALS))

COMMON_FLAGS := -O3 -march=native -ffast-math -DHWY_COMPILE_ALL_ATTAINABLE

GCC_FLAGS   := $(COMMON_FLAGS) -fopenmp
CLANG_FLAGS := $(COMMON_FLAGS) -fopenmp
ICX_FLAGS   := -O3 -xHost -ffast-math -qopenmp -DHWY_COMPILE_ALL_ATTAINABLE


gcc-spmv:
	cmake -B $(BUILD_DIR) \
		-DCMAKE_CXX_COMPILER=g++ \
		-DCMAKE_CXX_FLAGS="$(GCC_FLAGS) -I$(HWY_PATH)"
	cmake --build $(BUILD_DIR)
	./$(BUILD_DIR)/vectorization $(RUN_ARGS) gcc

clang-spmv:
	cmake -B $(BUILD_DIR) \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_CXX_FLAGS="$(CLANG_FLAGS) -I$(HWY_PATH)"
	cmake --build $(BUILD_DIR)
	./$(BUILD_DIR)/vectorization $(RUN_ARGS) clang

icx-spmv:
	@bash -c "source /opt/intel/oneapi/setvars.sh && \
	cmake -B $(BUILD_DIR) \
		-DCMAKE_CXX_COMPILER=icpx \
		-DCMAKE_CXX_FLAGS='$(ICX_FLAGS) -I$(HWY_PATH)' && \
	cmake --build $(BUILD_DIR) && \
	./$(BUILD_DIR)/vectorization $(RUN_ARGS) icx"


run:
	./$(BUILD_DIR)/vectorization $(RUN_ARGS)

	rm -rf $(BUILD_DIR)

%:
	@:

.PHONY: gcc-spmv clang-spmv icx-spmv run clean