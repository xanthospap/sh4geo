From `ROOT_DIR`:

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Build Options

- Build DEBUG version with sanitizer check: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON`; run the tests (see below) to utilize sanitizer checking.
- Build enabling SIMD operations `cmake -S . -B build -DDSO_SIMD=ON`.
- Build enabling pre-computation of a series of SH factors that include square roots (see `dso::CunninghamWeights`). This will most probably make the code faster but augment required memory resources (~1 to 1.5 Mb). To use this optimization, compile with `PRECOMPUTED_SQRT_SHFACS=ON`.

## Testing

example running tests on parallel: `ctest --test-dir build -j8 --output-on-failure`

## Benchmarks

### `axpy` for `MatrixStorageType::LwTriangularColWise`

- build the benchmark with no `DSO_SIMD`: 
```
cmake -S . -B build-simd-off -DCMAKE_BUILD_TYPE=Release -DDSO_SIMD=OFF -DENABLE_BENCHMARKS=ON
cmake --build build-simd-off -j

```
- build the benchmark with `DSO_SIMD`:
```
cmake -S . -B build-simd-on -DCMAKE_BUILD_TYPE=Release -DDSO_SIMD=ON -DENABLE_BENCHMARKS=ON
cmake --build build-simd-on -j
```

- compare:
```
./build-simd-off/bench_axpy_lwtricolwise
./build-simd-on/bench_axpy_lwtricolwise
```


### Precomputing SH (Cunningham) Weight

- build the benchmark with no `PRECOMPUTED_SQRT_SHFACS=[ON|OFF]` and then compare using the program  `bench/bench_sh2gradient`.