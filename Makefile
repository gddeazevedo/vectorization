BUILD_DIR := build
RUN_ARGS  := $(filter-out run gcc clang icx clean rebuild-gcc rebuild-clang rebuild-icx,$(MAKECMDGOALS))

GCC_FLAGS   := -O3 -march=native -fopenmp -ffast-math -funroll-loops -falign-loops=32 -falign-functions=32 -ftree-vectorize -fvect-cost-model=very-cheap
CLANG_FLAGS := -O3 -march=native -fopenmp -ffast-math
ICX_FLAGS   := -O3 -xHost -ffast-math -qopenmp

gcc-spmv: clean
	cmake -B $(BUILD_DIR) -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS="$(GCC_FLAGS)"
	cmake --build $(BUILD_DIR)
	./$(BUILD_DIR)/spmv_vectorization $(filter-out $@,$(MAKECMDGOALS)) gcc


clang-spmv: clean
	cmake -B $(BUILD_DIR) -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="$(CLANG_FLAGS)"
	cmake --build $(BUILD_DIR)

icx-spmv: clean
	cmake -B $(BUILD_DIR) -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS='$(ICX_FLAGS)' && \
	cmake --build $(BUILD_DIR)

run:
	@bash -c "source /opt/intel/oneapi/setvars.sh && ./$(BUILD_DIR)/spmv_vectorization $(RUN_ARGS)"

clean:
	rm -rf $(BUILD_DIR)

%:
	@:

.PHONY: gcc clang icx run clean