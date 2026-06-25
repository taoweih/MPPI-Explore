# Benchmarks

Benchmark scripts are run as Python modules from the repository root. Do not run
them by direct file path.

```bash
python3 -m benchmark.senior_thesis.scripts.u_point_mass_benchmark
python3 -m benchmark.sensitivity.scripts.u_point_mass_sensitivity
python3 -m benchmark.paper.scripts.u_point_mass_density_knn_v_kde
python3 -m benchmark.paper.scripts.go2_walk_tracking_error
```

Equivalent senior-thesis, sensitivity, and paper density scripts exist for
`ant` and `ur5e`.

## Layout

```text
benchmark/
  common/                 shared output-path and CLI helpers
  senior_thesis/scripts/  senior-thesis benchmark and plotting scripts
  senior_thesis/data/     local senior-thesis runs, plots, and saved weights
  sensitivity/scripts/    sensitivity sweep scripts
  sensitivity/data/       local sensitivity outputs
  paper/scripts/          paper-focused comparison scripts
  paper/data/             local paper benchmark outputs
```

Local outputs default to:

```text
benchmark/<category>/data/runs/<timestamped_run_dir>
```

Senior-thesis plots default to:

```text
benchmark/senior_thesis/data/plots/
```

To write outputs somewhere else, use either:

```bash
export MPPI_BENCH_OUTPUT_ROOT=/path/to/mppi_outputs
```

or pass:

```bash
--output-root /path/to/mppi_outputs
```

With an output root, runs are written under category subdirectories, for
example:

```text
/path/to/mppi_outputs/paper/runs/<timestamped_run_dir>
```

## Local Run

Activate the environment and run from the repository root:

```bash
conda activate mjwarp
cd /path/to/MPPI-Explore
```

For a short smoke test of the UPointMass KNN-vs-KDE paper benchmark:

```bash
python3 -m benchmark.paper.scripts.u_point_mass_density_knn_v_kde \
  --num-trials 1 \
  --max-iterations 200 \
  --parallel sequential \
  --num-gpus 1 \
  --freq-calibration-iters 0
```

For a normal local run on one GPU:

```bash
python3 -m benchmark.paper.scripts.u_point_mass_density_knn_v_kde \
  --parallel all \
  --max-workers 4 \
  --num-gpus 1 \
  --freq-calibration-iters 50
```

The Go2 tracking-error paper benchmark supports the same scheduling options,
but uses `--max-steps` instead of `--max-iterations`:

```bash
python3 -m benchmark.paper.scripts.go2_walk_tracking_error \
  --parallel all \
  --max-workers 4 \
  --num-gpus 1 \
  --freq-calibration-iters 50
```

If you want local outputs outside the Git checkout:

```bash
export MPPI_BENCH_OUTPUT_ROOT=/tmp/mppi_outputs

python3 -m benchmark.paper.scripts.u_point_mass_density_knn_v_kde \
  --parallel sequential \
  --num-gpus 1
```

## Paper Benchmark Parallelism

All scripts under `benchmark/paper/scripts/` support:

```text
--parallel sequential|controllers|axis|all
--max-workers <count>|auto
--num-gpus <count>
--freq-calibration-iters <count>
```

The parallel modes use one subprocess per controller/sweep point:

- `controllers`: batch controllers together for each sweep-axis value.
- `axis`: batch sweep-axis values together for each controller.
- `all`: submit every controller/sweep-point combination together.
- `sequential`: run in the main process without a separate calibration pass.

`--max-workers` limits concurrent subprocesses. `--num-gpus` assigns workers
round-robin across the GPUs visible through `CUDA_VISIBLE_DEVICES`.

For parallel modes, control-frequency measurements collected during the batch
are not used in final output. After both sweeps complete, the runner performs a
sequential frequency calibration with exclusive GPU access and overwrites the
frequency mean, standard deviation, and per-trial arrays. Set
`--freq-calibration-iters 0` to disable this final pass.

## Delta Cluster Run

Do not run GPU benchmarks on Delta login nodes. Clone or pull the repository in
your project space, then launch GPU work through Slurm.

```bash
ssh <netid>@dt-login.delta.ncsa.illinois.edu
cd /projects/<project_code>/$USER
git clone git@github.com:<you>/MPPI-Explore.git
cd MPPI-Explore
```

Keep generated outputs on project or scratch storage, not inside the Git
checkout:

```bash
export MPPI_BENCH_OUTPUT_ROOT=/work/hdd/<project_code>/$USER/mppi_outputs
mkdir -p "$MPPI_BENCH_OUTPUT_ROOT"
```

### Interactive Smoke Test

Request an interactive GPU node. Replace `<gpu_partition>` with the GPU
partition available to your allocation.

```bash
srun -A <project_code> \
  --partition=<gpu_partition> \
  --nodes=1 \
  --gpus-per-node=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=32g \
  --time=01:00:00 \
  --pty bash
```

Then run the shortened benchmark:

```bash
conda activate mjwarp
cd /projects/<project_code>/$USER/MPPI-Explore

python3 -m benchmark.paper.scripts.u_point_mass_density_knn_v_kde \
  --output-root "$MPPI_BENCH_OUTPUT_ROOT" \
  --num-trials 1 \
  --max-iterations 200 \
  --parallel sequential \
  --num-gpus 1 \
  --freq-calibration-iters 0
```

### Batch Run

Create a Slurm file such as `run_upm_knn_kde.slurm`:

```bash
#!/bin/bash
#SBATCH -A <project_code>
#SBATCH -J upm_knn_kde
#SBATCH -p <gpu_partition>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH -t 08:00:00
#SBATCH -o slurm-%x-%j.out

conda activate mjwarp
cd /projects/<project_code>/$USER/MPPI-Explore

export MPPI_BENCH_OUTPUT_ROOT=/work/hdd/<project_code>/$USER/mppi_outputs
export MPLCONFIGDIR=/tmp/mppi_matplotlib_$SLURM_JOB_ID
mkdir -p "$MPPI_BENCH_OUTPUT_ROOT" "$MPLCONFIGDIR"

python3 -m benchmark.paper.scripts.u_point_mass_density_knn_v_kde \
  --output-root "$MPPI_BENCH_OUTPUT_ROOT" \
  --num-gpus 2 \
  --parallel all \
  --max-workers 8 \
  --freq-calibration-iters 50 \
  --no-record-video
```

The equivalent Go2 command is:

```bash
python3 -m benchmark.paper.scripts.go2_walk_tracking_error \
  --output-root "$MPPI_BENCH_OUTPUT_ROOT" \
  --num-gpus 2 \
  --parallel all \
  --max-workers 8 \
  --freq-calibration-iters 50
```

Submit it:

```bash
sbatch run_upm_knn_kde.slurm
```

Remote outputs will be written to:

```text
/work/hdd/<project_code>/$USER/mppi_outputs/paper/runs/<timestamped_run_dir>
```

Set `--num-gpus` to match `#SBATCH --gpus-per-node`. Tune `--max-workers` based
on the number of GPUs, benchmark size, and how much CPU/GPU contention you see.

## Common Overrides

These options are available on senior-thesis and paper density benchmark
scripts:

```bash
--output-root PATH
--num-trials N
--max-iterations N
--parallel sequential|controllers|axis|all
--max-workers N
--num-gpus N
--freq-calibration-iters N
--record-video
--no-record-video
```

Sensitivity scripts support:

```bash
--output-root PATH
--num-trials N
--max-iterations N
```
