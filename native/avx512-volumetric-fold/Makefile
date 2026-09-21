CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -Wpedantic -std=c11
AVX512_FLAGS ?= -mavx512f -mavx512dq

.PHONY: all test clean windows-object

all: test_volumetric_fold

volumetric_fold.o: volumetric_fold.c volumetric_fold.h
	$(CC) $(CFLAGS) -c $< -o $@

volumetric_fold_avx512.o: volumetric_fold_avx512.S
	$(CC) $(CFLAGS) $(AVX512_FLAGS) -c $< -o $@

test_volumetric_fold.o: test_volumetric_fold.c volumetric_fold.h
	$(CC) $(CFLAGS) -c $< -o $@

test_volumetric_fold: volumetric_fold.o volumetric_fold_avx512.o test_volumetric_fold.o
	$(CC) $^ -o $@

test: test_volumetric_fold
	./test_volumetric_fold

windows-object:
	clang --target=x86_64-w64-windows-gnu $(AVX512_FLAGS) -c volumetric_fold_avx512.S -o volumetric_fold_avx512.win64.o

clean:
	rm -f *.o test_volumetric_fold
