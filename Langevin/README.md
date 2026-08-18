# numerical_integration — source overview and how to build/run

This directory contains the numerical integration code used in the project. The main responsibilities are:

- generating graphs (degree sequences and sparse interaction matrices),
- building ecosystems from interaction matrices,
- running deterministic and stochastic numerical integrators (Runge–Kutta, Milstein),
- utility helpers (random number generators, memory helpers, small math routines),
- several driver programs that create ensembles of graphs and run experiments producing summary files in `Results/`.

Files in this folder
--------------------

- `basic_toolbox.h` / `basic_toolbox.cpp`
  - General utilities: array helpers, memory allocation wrappers, RNG helpers (uniform, gaussian, Poisson, geometric), small math utilities (sum, max, search), file helpers, tiny wrappers to open files and create directories, and convenience macros.
  - Use: provides low-level building blocks used across the other modules.

- `ecosystem_numerical_integration_toolbox.h` / `ecosystem_numerical_integration_toolbox.cpp`
  - Builds and manipulates the ecosystem graph structure (sparse vertex/edge representation), read/write for interaction matrices (dense and sparse), extracting initial conditions, copying abundances, and a suite of numerical integrators.
  - Integrators include adaptive Cash–Karp Runge–Kutta, fixed RK4 and a Milstein integrator for demographic noise. There are driver functions tailored for phase-diagram runs (adaptive+static mixed driver), saving trajectories and summaries.
  - Provides top-level functions used by the main driver programs to evolve GLV dynamics on graphs.

- `generate_graph_from_sequence_toolbox.h` / `generate_graph_from_sequence_toolbox.cpp`
  - Implements utilities to build graphs from degree sequences. This includes routines to compute crossing index tables, Erdos–Gallai tests for graphicality, building reduced degree sequences, and finally a `graph_from_degree_sequence` function that constructs a (sparse) interaction matrix. It relies on Eigen `RealMatrix` for the interaction matrix representation.
  - Also contains save/load helpers for the sparse matrix format used by the rest of the codebase.

- Driver programs (examples):
  - `Ecosystem_Dynamics_on_Graphs_in_Temperature.cpp`
    - Purpose: generate graph(s) from degree sequences, build interaction matrices (potentially sparse), and evolve ecosystems in non-zero temperature (stochastic) setting. It uses the `partyally_asymmetric_gaussian` generator and saves extraction/matrix files and phase-diagram summaries.
    - Usage (from source):
      ```txt
      Usage: <binary> N c mu sigma epsilon T N_ext N_prev_ext N_meas
      ```

  - `Ecosystem_Dynamics_on_RRG_zero_Temperature.cpp`
    - Purpose: similar to the above but specialized for random-regular graphs (RRG) and zero temperature (deterministic dynamics). This program enforces `T == 0` and performs extractions, saves equilibrium points and summaries.
    - Usage:
      ```txt
      Usage: <binary> N c mu sigma epsilon T N_ext N_prev_ext N_meas
      ```

  - `Ecosystem_Dynamics_on_DirectedER_zero_Temperature.cpp`
    - Purpose: driver to extract directed Erdős–Rényi graphs (directed trivial interaction generator) and integrate dynamics at zero temperature. It enforces `T == 0`.
    - Usage:
      ```txt
      Usage: <binary> N c mu sigma T N_ext N_prev_ext N_meas
      ```

How the pieces fit together
---------------------------

- Graph generation: `generate_graph_from_sequence_toolbox` creates a sparse interaction matrix (Eigen `RealMatrix` is used internally). The program can save the sparse matrix to a textual file which is later read by the ecosystem loader.
- Ecosystem creation and evolution: `ecosystem_numerical_integration_toolbox` reads the matrix and per-vertex parameters (r, K, etc.), sets up the in-memory `graph` structure and runs time-integration using the selected integrator.
- Drivers: The `Ecosystem_Dynamics*` programs orchestrate extraction (many graphs), for each graph perform one or more measures (different initial conditions), run the integrators, and write summary files under `Results/` or per-extraction files under the created directories.

Build instructions
------------------

A `Makefile` is provided in this directory (`numerical_integration/Makefile`) to build the driver binaries. From the repository root you can run:

```bash
# build using the provided Makefile
make -C numerical_integration -j2
```

The Makefile defaults to using `numerical_integration/eigen-3.4.0` as the Eigen include directory; if your Eigen headers are installed elsewhere pass `EIGEN_INC`:

```bash
make -C numerical_integration EIGEN_INC=/path/to/eigen -j2
```

If you prefer manual compilation the commands below are equivalent examples — they are provided for reference only.

```bash
# from repository root (reference only)
mkdir -p build/bin
# adjust EIGEN_INC to your Eigen install path
EIGEN_INC=/path/to/eigen-3.4.0
g++ -std=c++17 -O2 -I${EIGEN_INC} \
  numerical_integration/basic_toolbox.cpp \
  numerical_integration/generate_graph_from_sequence_toolbox.cpp \
  numerical_integration/ecosystem_numerical_integration_toolbox.cpp \
  numerical_integration/Ecosystem_Dynamics_on_Graphs_in_Temperature.cpp \
  -o build/bin/Ecosystem_Dynamics_on_Graphs_in_Temperature \
  -lm

# Build the other driver variants similarly (replace final cpp file and output name):
g++ -std=c++17 -O2 -I${EIGEN_INC} \
  numerical_integration/basic_toolbox.cpp \
  numerical_integration/generate_graph_from_sequence_toolbox.cpp \
  numerical_integration/ecosystem_numerical_integration_toolbox.cpp \
  numerical_integration/Ecosystem_Dynamics_on_RRG_zero_Temperature.cpp \
  -o build/bin/Ecosystem_Dynamics_on_RRG_zero_Temperature -lm

g++ -std=c++17 -O2 -I${EIGEN_INC} \
  numerical_integration/basic_toolbox.cpp \
  numerical_integration/generate_graph_from_sequence_toolbox.cpp \
  numerical_integration/ecosystem_numerical_integration_toolbox.cpp \
  numerical_integration/Ecosystem_Dynamics_on_DirectedER_zero_Temperature.cpp \
  -o build/bin/Ecosystem_Dynamics_on_DirectedER_zero_Temperature -lm
```

Notes and caveats
-----------------

- Eigen: the header included in `generate_graph_from_sequence_toolbox.h` points to `/path/to/eigen-3.4.0/Eigen/Dense`. Update `-I` accordingly or change that include to match your local Eigen install.
- Some source files in this folder reference additional helper headers and rely on the macros and memory helpers provided by `basic_toolbox.h`.
- A number of functions and blocks in the provided sources are heavy numerical code (adaptive stepsize logic, Milstein implementation, file IO). The README summarizes purpose and usage but does not expand the full algorithmic details.
- Several files create and remove directories and temporary files via `system("mkdir ...")` and `system("rm ...")` — ensure you run the programs in a directory where those operations are acceptable.



API reference (concise)
-----------------------

Below is a compact list of the main exported functions (from the headers) with a one-line purpose. This is intended as a quick developer reference — inspect the headers for full prototypes and comments.

basic_toolbox (utilities)
- compare_desc(const void*, const void*): comparator for qsort (descending).
- my_max(int,int): integer max.
- sum_array(double*,int) / sum_array(int*,int): sums elements.
- max_array(int*,int): maximum of int array.
- loadArray(FILE*,int*,int): read ints from file.
- fill_array(int*,int,int): fill array with a value.
- are_arrays_equal(...): compare double arrays with tolerance.
- initialize_double_array(...), equalize_double_array(...), combine_double_array(...): common array ops.
- RNG(), RNG_0(), RNG_1(), UNG(a,b), GNG(mean,std), Positive_GNG(mean,std): random generators.
- Init_Poisson(...), poisson_RNG(...), Init_PowerLaw(...), Init_Geometric(...), Geometric_RNG(...): RNG/table init helpers.
- my_double_malloc/calloc/realloc, my_int_malloc/calloc/realloc, my_char_malloc/realloc: allocation helpers with CHECK_ALLOC usage.
- my_open_writing_file / my_open_appending_file / my_open_reading_file: safe fopen wrappers.
- my_create_directory: create directory helper.

ecosystem_numerical_integration_toolbox (ecosystem & integrators)
- ecosystem_initialization(int): allocates and returns empty ecosystem structure.
- graph_extraction(...): generic graph extraction routine for dense/sparse graphs.
- directed_ER_ecosystem_extraction(...): extract directed Erdős–Rényi ecosystem.
- symmetric_gaussian_extraction(...), partyally_asymmetric_gaussian(...): interaction weight generators.
- save_dense_interaction_matrix_to_file / save_sparse_interaction_matrix_to_file: exporters.
- save_ecosystem_to_file / load_ecosystem_from_file: persistence helpers.
- extract_and_save_random_initial_conditions / load_initial_conditions: initial condition IO.
- GLV_equation / log_LV_gen_eq: compute GLV derivatives.
- rkck_algorithm, rk5_quality_control_stepper, log_RungeKutta_adaptive_driver, rk4_algorithm: Runge–Kutta integrators (adaptive and fixed).
- log_RungeKutta_adaptive_mixed_static_driver_for_phase_diagram: mixed integration driver used by phase-diagram runs.
- Milstein_onestep_GLV_demographic_noise and Milstein_driver_*: stochastic Milstein integrators and driver wrappers.
- check_ecosystem_divergence / check_ecosystem_equilibrium_zero_temperature / check_ecosystem_thermal_equilibrium: diagnostics.

generate_graph_from_sequence_toolbox (graph-from-sequence)
- save_matrix_to_file_sparse_ijaij / save_matrix_to_file_sparse_ijaijaji / load_matrix_to_file_sparse_ijaij: sparse matrix persistence.
- build_crossing_index_table* variants: helpers for the Erdos–Gallai crossing index table.
- erdos_gallai_test: test if degree sequence is graphical.
- build_dprime / update_rds_connection / update_rdsp_connection / update_rdsp_last_connection: internal helpers for sequence rewiring.
- graph_from_degree_sequence(...): main function that constructs a sparse RealMatrix graph from degree sequence and weight generator.
