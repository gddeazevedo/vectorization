# Vectorization Benchmarks

Benchmarks de operações de algebra linear com vetorização (AVX256, AVX512) e paralelismo (OpenMP).

## Operacoes disponiveis

- **spmv** — Sparse Matrix-Vector Multiplication (SpMV) com BCSR
- **ilu** — Incomplete LU Factorization (em desenvolvimento)

## Dependencias

- GCC (g++) com suporte a OpenMP
- Intel oneAPI (icpx) — opcional
- CMake >= 3.10
- Google Highway (SIMD)

Todas as dependencias sao instaladas automaticamente via Docker.

## Setup

```bash
make up      # sobe o container
make bash    # acessa o shell do container
```

Para derrubar o container:

```bash
make down
```

## Compilar e rodar benchmarks

Dentro do container:

```bash
make <compilador> <operacao>
```

### Exemplos

```bash
make gcc spmv    # compila com g++ e roda benchmark SpMV
make icx spmv    # compila com icpx e roda benchmark SpMV
```

## Estrutura

```
src/            # ponto de entrada (main.cpp)
libs/
  cli/          # CLI de selecao de operacao
  bcsr/         # formato Blocked CSR
  bc_matvec/    # variantes de matvec (base, AVX256, AVX512, OpenMP, Highway)
  benchmarks/
    spmv/       # benchmarks de SpMV
    ilu/        # benchmarks de ILU (TODO)
  ilu/          # fatoracao ILU (TODO)
experiments/    # resultados dos benchmarks (CSV)
plots/          # scripts de visualizacao
demos/          # exemplos avulsos de vetorizacao
```
