# AI Models Benchmarking and Resource Monitoring Tools for Windows and Linux

Includes a general-purpose CLI resource monitor (`resource_monitor.py`), llama.cpp benchmarking orchestrators (`mass_llama_bench.py`, `mass_llama_server_benchy.py`, `mass_benchmark_embeddings.py`, `mass_benchy.py`), a Lemonade server benchmarking orchestrator (`mass_lemonade_benchy.py`), and a plotting tool (`plot_json_benchmarks.py`).

> **Note:** This toolset supports both Windows and Linux environments. 

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)

  * [Resource Monitor](#resource-monitor)
  * [Mass LLaMA Bench](#mass-llama-bench)
  * [Mass LLaMA Server Benchy](#mass-llama-server-benchy)
  * [Mass Benchmark Embeddings](#mass-benchmark-embeddings)
  * [Mass Lemonade Benchy](#mass-lemonade-benchy)
  * [Mass Benchy](#mass-benchy)
  * [Plot JSON Benchmarks](#plot-json-benchmarks)

* [JSON File Naming Convention](#json-file-naming-convention)
* [Known Issue with Monitoring NPU Memory on Linux](#known-issue-with-monitoring-npu-memory-on-linux)

## Features

* 📊 **Resource Monitoring**: Real-time CPU, RAM, GPU, and NPU utilization tracking during execution
* 🐧 **Cross-Platform**: Runs on both Windows and Linux, using PDH counters on Windows and DRM/sysfs (`/sys/class/accel`, `/sys/class/drm`) telemetry on Linux for GPU and NPU monitoring
* 🚀 **Batch Benchmarking**: Run benchmarks across multiple models, installations, endpoints, and configurations
* 🎯 **Flexible Configuration**: Support for multiple option sets to test different configurations
* 🖥️ **Server Management**: Launch and manage llama-server instances automatically (`mass_llama_server_benchy.py`)
* 🌐 **External Endpoints**: Benchmark existing endpoints without server management (`mass_benchy.py`)
* 📁 **Structured Output**: JSON-formatted results with both command output and resource statistics
* 📊 **Plotting**: Generates plots and overlay comparisons from JSON results (`plot_json_benchmarks.py`)

## Prerequisites

### Required Software

* **Python**: 3.10 or later (platform-specific build required: x64 for Intel/AMD systems, ARM64 for ARM-based systems)
* **Operating System**: Windows 10/11 or Linux
* **Linux power mode support (optional)**: `powerprofilesctl` (from `power-profiles-daemon`) is required if you want `-p/--power-mode` to change Linux power modes

### Required Python Packages

* `psutil` - For CPU and RAM monitoring
* `pywin32` — For Windows GPU monitoring via PDH (Windows only; not required on Linux)
* `matplotlib` — For generating PDF plots
* `json-repair` — For robust parsing of llama-bench JSON output
* `plotly` — For generating interactive HTML plots
* `llama-benchy` — CLI required for some benchmarking workflows.

### Hugging Face Token for `llama-benchy` Chat Completions

When benchmarking OpenAI-compatible `v1/chat/completions` endpoints with scripts that invoke `llama-benchy` (for example, `mass_llama_server_benchy.py`, `mass_lemonade_benchy.py`, and `mass_benchy.py`), set a Hugging Face token in the `HF_TOKEN` environment variable.
1. Create or sign in to your Hugging Face account: `https://huggingface.co/`
2. Go to **Settings** → **Access Tokens**: `https://huggingface.co/settings/tokens`
3. Create a token with at least read-only access.
4. Set the token in your shell before running benchmark scripts:

   * Windows (PowerShell): `$env:HF_TOKEN = "<your_token_here>"`
   * Windows (CMD): `set HF_TOKEN=<your_token_here>`
   * Linux/macOS (bash): `export HF_TOKEN="<your_token_here>"`

## Installation

### Clone and enter the repository

git clone https://github.com/HPInc/ai-models-performance-measurement-suite

cd ai-models-performance-measurement-suite

### One-step setup

#### Windows

python -m pip install -r windows_requirements.txt

#### Linux

python3 -m venv .venv

source .venv/bin/activate

python3 -m pip install -r linux_requirements.txt

### Verify Installation

`python mass_llama_bench.py --help`

## Usage

### Resource Monitor

`resource_monitor.py` is a standalone tool that collects system resource statistics (RAM, CPU, GPU utilization and memory, and NPU memory) while running **any** command-line process. It writes JSON output for later plotting with other tools.

#### Command-Line Arguments

* `-s, --sample-interval`: Seconds between resource monitoring samples (range: 0.2-1.0, default: 0.2)
* `-o, --output-dir`: Optional directory to write JSON results
* `-e, --echo-stderr`: Echo stderr to console instead of capturing
* `-n, --normalize-resource-data`: Normalize resource data by subtracting prestart values from each data point
* `-p, --power-mode`: Set power mode before running the command (Windows supported; Linux requires `powerprofilesctl`; choices: `performance`, `balanced`, `power-saver`)
* `--cmd`: Shell command string to execute (alternative to using `--` remainder)

#### Output JSON File Structure
```
{
   "command": <command that was executed>,
   "result:" <STDOUT and STDERR output from the command>,
   "stats": <JSON dictionary of runtime statistics (CPU, RAM, GPU, NPU usage)>,
   "system_info": <JSON dictionary of system information. Content is OS dependent.>
}
```

#### Monitoring background processes with `waiter.py`

If you want to measure “what the system is doing” while *you* perform actions (open apps, copy files, run a GUI workload, etc.), use `waiter.py` as a long-running placeholder command:

1. Start monitoring:

   `python resource_monitor.py -o .\output -- python waiter.py`

2. Perform the background activity you want to measure.
3. Press any key in the `waiter.py` console window to stop and finalize the captured data.

#### Examples

Monitor `systeminfo` and write JSON results into an output directory:

`python resource_monitor.py -o .\output -- systeminfo`

Monitor a background/idle system state using `waiter.py` (waits until you press a key):

`python resource_monitor.py -o .\output -- python waiter.py`

### Mass LLaMA Bench

`mass_llama_bench.py` orchestrates running `llama-bench` across multiple llama.cpp installations, LLM models, and llama-bench parameter sets, while monitoring system resources during each benchmark run.

#### Runtime Prerequisites

1. At least one .gguf model file (`-m`)
2. At least one llama.cpp installation with `llama-bench` (`-l`)

#### Command-Line Arguments

##### Required Arguments

* `-o, --output-dir`: Output directory to store JSON and PDF files (created if doesn't exist)
* `-m, --model`: Path to a GGUF model file (can be specified multiple times)

##### Optional Arguments

* `-l, --llamacpp-dir`: Path to llama.cpp installation directory (default: current working directory; can be specified multiple times). Each additional `-l` creates a separate benchmark run.
* `-e, --extra-options`: Additional options to pass to llama-bench, quoted (can be specified multiple times). Each additional `-e` creates a separate benchmark run.
* `-f, --fixed-options`: Fixed options to pass to llama-bench for all runs, quoted. Unlike `-e`, this does not create additional benchmark runs; the options are applied to every run. Only one `-f` parameter is allowed.
* `-s, --sample-interval`: Seconds between resource monitoring samples (range: 0.2-1.0, default: 0.2)
* `-r, --runs`: Number of full repeated runs per configuration (default: `1`)
* `-p, --power-mode`: Set power mode before running benchmarks (Windows supported; Linux requires `powerprofilesctl`; choices: `performance`, `balanced`, `power-saver`; can be specified multiple times to create separate runs)
* `-d, --description`: Custom description for the benchmark run, stored in convenience_metrics and used as a label differentiator in plots (default: `llama-bench`)
* `--reset-environment`: Reset benchmark environment (best effort) before each llama-bench run
* `-v, --verbose`: Enable verbose logging with debug output and append `-v` to the underlying `llama-bench` command

#### Output JSON File Structure

```
{
   "llama_bench_results": <llama-bench results from STDOUT formatted as a JSON dictionary>,
   "convenience_metrics": <JSON dictionary of convenience metrics>,
   "runtime_stats": <JSON dictionary of runtime statistics (CPU, RAM, GPU, NPU usage)>,
   "system_info": <JSON dictionary of system information. Content is OS dependent.>
}
```

#### Examples

##### Example 1: Single Model, Single Installation

`python mass_llama_bench.py -o .\output -m \models\gpt-oss-20B.gguf -l \llama-cpp-dir`

##### Example 2: Multiple Installations with Options

`python mass_llama_bench.py -o \json\testrun  -m \models\gpt-oss-20B.gguf -l \llama-b6876-bin-win-vulkan-x64 -l \llama-b6877-bin-win-cuda-x64 -e "-fa 0" -e "-fa 1"`

This runs benchmarks with flash attention disabled (`-fa 0`) and enabled (`-fa 1`), creating separate test runs for each option combination.

##### Example 3: Fixed Option Applied to All Runs

`python mass_llama_bench.py -o .\results -l \llama-cpp-dir -m \models\gpt-oss-20B.gguf -f "-ngl 99" -e "-fa 0" -e "-fa 1"`

This applies 99 GPU layers (`-ngl 99`) to all runs, while creating separate runs for flash attention disabled and enabled.

### Mass LLaMA Server Benchy

`mass_llama_server_benchy.py` orchestrates running `llama-server` and `llama-benchy` across multiple llama.cpp installations, LLM models, server configurations, and llama-benchy parameter sets. It manages the llama-server lifecycle (start, capture URL, shutdown) for each benchmark run and monitors system resources during each run.

#### Runtime Prerequisites

* One or more `llama.cpp` installations with `llama-server` executable (`-l`).
* `llama-benchy` executable available in PATH or callable environment.
* One or more GGUF LLM files (`-m`).

#### Command-Line Arguments

##### Required Arguments

* `-o, --output-dir`: Output directory to store JSON files (created if doesn't exist)
* `-m, --model`: Path to a GGUF model file, optionally with a comma-separated model name for llama-benchy (format: `path/to/model.gguf` or `path/to/model.gguf,model-name-for-benchy`; can be specified multiple times). If omitted, the model must be specified via `-c`, `-s`, or `-j` options.

##### Optional Arguments

* `-l, --llamacpp-dir`: Path to llama.cpp installation directory containing llama-server (default: current working directory; can be specified multiple times)
* `-c, --constant-server-parms`: Options to pass to llama-server for all runs, quoted. Only one `-c` parameter is allowed.
* `-s, --server-parms`: Additional options to pass to llama-server, quoted. Multiple `-s` parameters create separate server-option configurations.
* `-e, --extra-options`: Additional options to pass to llama-benchy, quoted (can be specified multiple times to create separate runs; each run starts and stops a fresh llama-server instance)
* `-f, --fixed-options`: Fixed options to pass to llama-benchy for all runs, quoted. Only one `-f` parameter is allowed.
* `-j, --joint-options`: Combined server and benchy options in one string, separated by a comma. The part before the comma is passed to llama-server; the part after the comma is passed to llama-benchy. If no comma is present, the entire string is treated as llama-benchy options. Multiple `-j` parameters create separate benchmark runs. Example: `-j "-ngl 99,--pp 128"`
* `-i, --sample-interval`: Seconds between resource monitoring samples (default: 0.2)
* `-r, --runs`: Number of full repeated runs per configuration (default: `1`)
* `-p, --power-mode`: Set power mode before running benchmarks (Windows supported; Linux requires `powerprofilesctl`; choices: `performance`, `balanced`, `power-saver`; can be specified multiple times)
* `-d, --description`: Custom description for the benchmark run, stored in convenience_metrics and used as a label differentiator in plots (default: `llama-server`)
* `--reset-environment`: Reset benchmark environment (best effort) before each server launch
* `-v, --verbose`: Enable verbose logging with debug output and append `-v` to the underlying `llama-server` command

#### Output JSON File Structure

```
{
   "llama_benchy_results": <llama-benchy results from STDOUT formatted as a JSON dictionary>,
   "convenience_metrics": <JSON dictionary of convenience metrics>,
   "runtime_stats": <JSON dictionary of runtime statistics (CPU, RAM, GPU, NPU usage)>,
   "system_info": <JSON dictionary of system information. Content is OS dependent.>
}
```

#### Examples

##### Example 1: Single Model with Server

`python mass_llama_server_benchy.py -m \models\model.gguf -l \llama-dir -o \json\testrun`

##### Example 2: Server Model with Separate Benchy Model Name

`python mass_llama_server_benchy.py -m "\models\model.gguf,my-model-name" -l \llama-dir -o \json\testrun`

##### Example 3: Constant and Variable Server Options

`python mass_llama_server_benchy.py -m \models\model.gguf -l \llama-dir -o \json\testrun -c "-ngl 99" -s "-fa 0" -s "-fa 1"`

##### Example 4: Multiple Benchy Option Sets

`python mass_llama_server_benchy.py -m \models\model.gguf -l \llama-dir -o \json\testrun -e "--pp 128" -e "--pp 256"`

### Mass Benchmark Embeddings

`mass_benchmark_embeddings.py` orchestrates running `llama-server` and `benchmark_embeddings.py` across multiple models and server/benchmark option sets. It manages the llama-server lifecycle for each benchmark run, targets the `/v1/embeddings` endpoint, and records resource monitoring data for each run.

#### Runtime Prerequisites

* One or more `llama.cpp` installations with `llama-server` executable (`-l`).
* One or more GGUF embeddings files (`-m`).

#### Command-Line Arguments

##### Required Arguments

* `-o, --output-dir`: Output directory to store JSON files (created if doesn't exist)
* `-m, --model`: Path to a GGUF model file to pass to llama-server (can be specified multiple times). If omitted, the model must be specified via `-c`, `-s`, or `-j` options.

##### Optional Arguments

* `-l, --llamacpp-dir`: Path to llama.cpp installation directory containing llama-server.exe (default: current working directory; can be specified multiple times)
* `-c, --constant-server-parms`: Options to pass to llama-server for all runs, quoted. Only one `-c` parameter is allowed.
* `-s, --server-parms`: Additional options to pass to llama-server, quoted. Multiple `-s` parameters create separate server-option configurations.
* `-e, --extra-options`: Additional options to pass to `benchmark_embeddings.py`, quoted (can be specified multiple times to create separate runs; each run starts and stops a fresh llama-server instance)
* `-f, --fixed-options`: Fixed options to pass to `benchmark_embeddings.py` for all runs, quoted. Only one `-f` parameter is allowed.
* `-j, --joint-options`: Combined server and benchmark options in one string, separated by a comma. The part before the comma is passed to llama-server; the part after the comma is passed to `benchmark_embeddings.py`. If no comma is present, the entire string is treated as benchmark options. Multiple `-j` parameters create separate benchmark runs. Example: `-j "-ngl 99,--samples 1000"`
* `-i, --sample-interval`: Seconds between resource monitoring samples (default: 0.2)
* `-r, --runs`: Number of full repeated runs per configuration (default: `1`)
* `-p, --power-mode`: Set power mode before running benchmarks (Windows supported; Linux requires `powerprofilesctl`; choices: `performance`, `balanced`, `power-saver`; can be specified multiple times)
* `-d, --description`: Custom description for the benchmark run, stored in convenience_metrics and used as a label differentiator in plots (default: `benchmark-embeddings`)
* `--reset-environment`: Reset benchmark environment (best effort) before each server launch
* `-v, --verbose`: Enable verbose logging with debug output and append `-v` to the underlying `llama-server` command

#### Output JSON File Structure

```
{
   "embedding_bench_results": <embeddings benchmark results from STDOUT formatted as a JSON dictionary>,
   "convenience_metrics": <JSON dictionary of convenience metrics>,
   "runtime_stats": <JSON dictionary of runtime statistics (CPU, RAM, GPU, NPU usage)>,
   "system_info": <JSON dictionary of system information. Content is OS dependent.>
}
```

#### Examples

##### Example 1: Single Model

`python mass_benchmark_embeddings.py -m \models\model.gguf -l \llama-dir -o \json\testrun`

##### Example 2: Constant and Variable Server Options

`python mass_benchmark_embeddings.py -m \models\model.gguf -l \llama-dir -o \json\testrun -c "-ngl 99" -s "-fa 0" -s "-fa 1"`

##### Example 3: Multiple benchmark_embeddings Option Sets

`python mass_benchmark_embeddings.py -m \models\model.gguf -l \llama-dir -o \json\testrun -e "--samples 500" -e "--samples 2000"`

### Mass Lemonade Benchy

`mass_lemonade_benchy.py` orchestrates running `lemonade-server` and `llama-benchy` across multiple models, server configurations, and llama-benchy parameter sets. It manages the lemonade model lifecycle (load model, run benchmark, unload model) for each benchmark run and monitors system resources including NPU memory during each run.

#### Runtime Prerequisites

* `lemonade` CLI installed and available in PATH.
* Running lemonade server service.
* `llama-benchy` executable available in PATH or callable environment.
* One or more GGUF embeddings files (`-m`).

#### Command-Line Arguments

##### Required Arguments

* `-o, --output-dir`: Output directory to store JSON files (created if doesn't exist)
* `-m, --model`: Model name for lemonade server, optionally with a comma-separated model name for llama-benchy (format: `ModelName` or `ModelName,model-name-for-benchy`; can be specified multiple times)

##### Optional Arguments

* `--host`: Lemonade server host address (default: `127.0.0.1`)
* `--port`: Lemonade server port number (default: `13305`)
* `-c, --constant-server-parms`: Options to pass to lemonade load for all runs, quoted. Only one `-c` parameter is allowed.
* `-s, --server-parms`: Additional options to pass to lemonade load, quoted. Multiple `-s` parameters create separate model load/unload cycles.
* `-e, --extra-options`: Additional options to pass to llama-benchy, quoted (can be specified multiple times to create separate runs)
* `-f, --fixed-options`: Fixed options to pass to llama-benchy for all runs, quoted. Only one `-f` parameter is allowed.
* `-j, --joint-options`: Combined server and benchy options in one string, separated by a comma. The part before the comma is passed to lemonade load; the part after the comma is passed to llama-benchy. If no comma is present, the entire string is treated as llama-benchy options. Multiple `-j` parameters create separate benchmark runs. Example: `-j "--llamacpp vulkan,--pp 128"`
* `-i, --sample-interval`: Seconds between resource monitoring samples (default: 0.2)
* `-p, --power-mode`: Set power mode before running benchmarks (Windows supported; Linux requires `powerprofilesctl`; choices: `performance`, `balanced`, `power-saver`; can be specified multiple times)
* `-d, --description`: Custom description for the benchmark run, stored in convenience_metrics and used as a label differentiator in plots (default: `lemonade_server`)
* `--reset-environment`: Reset benchmark environment (best effort) before each model load
* `-v, --verbose`: Enable verbose logging with debug output

#### Output JSON File Structure

```
{
   "llama_benchy_results": <llama-benchy results from STDOUT formatted as a JSON dictionary>,
   "convenience_metrics": <JSON dictionary of convenience metrics>,
   "runtime_stats": <JSON dictionary of runtime statistics (CPU, RAM, GPU, NPU usage)>,
   "system_info": <JSON dictionary of system information. Content is OS dependent.>
}
```
#### Examples

##### Example 1: Single Model

`python mass_lemonade_benchy.py -m ModelName -o \json\testrun`

##### Example 2: Model with Separate Benchy Model Name

`python mass_lemonade_benchy.py -m "ModelName,my-model-name" -o \json\testrun`

##### Example 3: Constant and Variable Server Options

`python mass_lemonade_benchy.py -m ModelName -o \json\testrun -c "--llamacpp vulkan" -s "--context-length 2048" -s "--context-length 4096"`

##### Example 4: Multiple Power Modes (Windows or Linux with `powerprofilesctl`)

`python mass_lemonade_benchy.py -m ModelName -o \json\testrun -p performance -p balanced`

### Mass Benchy

`mass_benchy.py` runs `llama-benchy` against one or more external endpoints that are already running and reachable. Unlike `mass_llama_server_benchy.py`, it does **not** launch or manage `llama-server`.

#### Runtime Prerequisites

* `llama-benchy` executable available in PATH or callable environment.
* One or more already running and reachable inference endpoints.

#### Quick Start

`python mass_benchy.py -u http://localhost:8080 -o \json\testrun`

#### Command-Line Arguments

##### Required Arguments

* `-o, --output-dir`: Output directory to store JSON files (created if doesn't exist)
* `-u, --url`: URL of an endpoint to benchmark (can be specified multiple times to create separate runs)

##### Optional Arguments

* `-m, --model`: Model name to pass to llama-benchy
* `-e, --extra-options`: Additional options to pass to llama-benchy, quoted (can be specified multiple times to create separate runs)
* `-f, --fixed-options`: Fixed options to pass to llama-benchy for all runs, quoted. Only one `-f` parameter is allowed.
* `-i, --sample-interval`: Seconds between resource monitoring samples (default: 0.2)
* `-p, --power-mode`: Set power mode before running benchmarks (Windows supported; Linux requires `powerprofilesctl`; choices: `performance`, `balanced`, `power-saver`; can be specified multiple times)
* `-d, --description`: Custom description for the benchmark run, stored in convenience_metrics and used as a label differentiator in plots (default: `endpoint`)
* `-v, --verbose`: Enable verbose logging with debug output

#### Output JSON File Structure

```
{
   "llama_benchy_results": <llama-benchy results from STDOUT formatted as a JSON dictionary>,
   "convenience_metrics": <JSON dictionary of convenience metrics>,
   "runtime_stats": <JSON dictionary of runtime statistics (CPU, RAM, GPU, NPU usage)>,
   "system_info": <JSON dictionary of system information. Content is OS dependent.>
}
```

#### Examples

##### Example 1: Single Endpoint

`python mass_benchy.py -u http://localhost:8080 -o \json\testrun`

##### Example 2: Multiple Endpoints

`python mass_benchy.py -u http://host1:8080 -u http://host2:8080 -o \json\testrun`

##### Example 3: Model Name with Multiple Option Sets

`python mass_benchy.py -u http://localhost:8080 -m my-model -o \json\testrun -e "--pp 128" -e "--pp 256"`

### Plot JSON Benchmarks

`plot_json_benchmarks.py` generates overlay plots from JSON benchmark result files produced by any of the benchmarking scripts, including resource_monitor.py. It supports both PDF output (via matplotlib) and interactive HTML output (via Plotly).

#### Command-Line Arguments

* `-o, --output-dir`: Directory where plot files will be written (required)
* `-i, --input-directory`: Directory containing JSON benchmark files (can be specified multiple times)
* `-j, --json-file`: Path to a specific JSON benchmark file (can be specified multiple times)
* `-p, --prefix`: Prefix for output plot filenames (default: `benchmark`)
* `-l, --label-sort`: Sort bars in all bar charts by label instead of metric value
* `-b, --browser-plots`: Generate interactive HTML plots using Plotly instead of PDF
* `-n, --normalize-resource-data`: Normalize resource data by subtracting prestart values
* `-v, --verbose`: Print verbose output

The following plots are always generated:

* **Resource line plots**: Usage/utilization over time line plots.
* **Memory usage bar chart**: Compares average and maximum system RAM usage across benchmark runs
* **GPU memory usage bar chart**: Compares average and maximum GPU memory usage across benchmark runs
* **CPU utilization bar chart**: Compares average and maximum CPU utilization across benchmark runs
* **GPU utilization bar chart**: Compares average and maximum GPU utilization across benchmark runs
* **NPU memory usage bar chart**: Compares average and maximum NPU memory usage across benchmark runs

The following plots are generated by mass_llama_bench.py, mass_llama_server_benchy.py, mass_lemonade_benchy.py, and mass_benchy.py:

* **TTFT bar chart**: Compares Time To First Token across benchmark runs
* **avg\_ts bar chart**: Compares average tokens per second across benchmark runs

The following plots are generated by mass_benchmark_embeddings.py:

* **Embedding throughput**: Samples per second
* **P50 batch latency**: Median time for processing a batch of text samples

Plot legend labels are automatically differentiated based on fields that vary between datasets (model, build version, options, power mode, hostname, run label, variant, etc.).

Plot files are saved as PDFs by default, or as interactive HTML files when using the `-b` flag.

#### Examples

##### Example 1: Generate PDF Plots from a Directory

`python plot_json_benchmarks.py -i \path\to\json\files -o \output\dir`

##### Example 2: Generate Interactive HTML Plots

`python plot_json_benchmarks.py -i \path\to\json\files -o \output\dir -b`

##### Example 3: Specific JSON Files with Custom Prefix

`python plot_json_benchmarks.py -j file1.json -j file2.json -o \output\dir -p comparison`

##### Example 4: Sort Bar Charts by Label

`python plot_json_benchmarks.py -i \path\to\json\files -o \output\dir -l`

## JSON File Naming Convention

JSON output file names are prefixed by the script that generated them:

* `llama-bench_` — from `mass_llama_bench.py`
* `llama-server_` — from `mass_llama_server_benchy.py`
* `emb-server_` — from `mass_benchmark_embeddings.py`
* `lemonade_server_` — from `mass_lemonade_benchy.py`
* `endpoint_` — from `mass_benchy.py`

The remainder of the filename is built from:

* Hostname
* Llama.cpp release tag (four digits) or model name
* Dashed options
* Power mode label (if applicable)

Examples:
* `llama-bench_MYHOST_b7818_gpt-oss-20B.gguf_-fa_0_Balanced.json`
* `llama-server_MYHOST_b9048_gpt-oss-20B.gguf_Performance.json`
* `emb-server_MYHOST_b9048_emb-model.gguf_Efficiency.json`
* `lemonade_server_MYHOST_ModelName_Performance.json`
* `endpoint_MYHOST_localhost_8080_Balanced.json`

### Known Issue with Monitoring NPU Memory on Linux

You may need to run with elevated privileges to access `/sys/class/accel` for NPU memory monitoring on Linux. If your NPU memory measurements are all zero, and whatever you are doing uses the NPU, consider running with 'sudo env PATH="$PATH"'.

## Contact

**Maintainer:** "Paul Thomas" <surfidaho3877@gmail.com>

---
*Built for systematic llama.cpp performance evaluation*
