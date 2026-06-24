# MPPI-Explore

## Setup

```bash
conda env create -f environment.yml
conda activate mjwarp
```

Requires an NVIDIA GPU with CUDA support.

## Benchmarks

Run benchmark scripts as modules from the repo root:

```bash
python3 -m benchmark.senior_thesis.scripts.u_point_mass_benchmark
python3 -m benchmark.sensitivity.scripts.u_point_mass_sensitivity
python3 -m benchmark.paper.scripts.u_point_mass_density_knn_v_kde
python3 -m benchmark.paper.scripts.go2_walk_tracking_error
```

Equivalent senior-thesis, sensitivity, and paper density scripts exist for
`ant` and `ur5e`.
Local outputs default to `benchmark/<category>/data/runs/`; senior-thesis
plots default to `benchmark/senior_thesis/data/plots/`.

For Delta or other cluster runs, keep generated data on scratch/project storage:

```bash
export MPPI_BENCH_OUTPUT_ROOT=/path/to/project/mppi_outputs
python3 -m benchmark.senior_thesis.scripts.u_point_mass_benchmark --num-gpus 2
```

You can also pass `--output-root /path/to/project/mppi_outputs` per run.
