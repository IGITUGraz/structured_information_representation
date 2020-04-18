# Source code for *A model for structured information representation in neural networks of the brain*

## Requirements

* Python >= 3.5
* NEST 2.12
* NumPy 1.17.0
* Matplotlib 3.1.1
* Scikit-learn 0.21.2

## Installing the NEST module

To build and install the required NEST module, run the *run-build.sh* script in
the *nest_module* subdirectory. This requires NEST to be set up properly with
all environment variables set.

## Generating the data

To generate the data for all the main experiments described in the results, run
*run_experiments.sh* (note that this can take very long, i.e. up to a day).
Afterwards, run *summary_experiments.sh* to get an overview of the results. To generate
just parts of the data, see *run_all.sh* for how to invoke the Python files.

## Running the robustness experiments

To generate the data for the random parameter variation experiments described
in the methods, run *run_robustness_experiments.sh* (note that this can take very long, i.e. multiple days).  Afterwards, run
*summary_robustness_experiments.sh* to get an overview of the results.
