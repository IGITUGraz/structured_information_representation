#!/bin/sh

# note: this expects a content space "cspace0" to have been
# generated, e.g. as done by run_experiments.sh

# generate parameter variations
BASE_CFG=final_config
NUM_CFG=100
CSPACE=cspace0

for STD in 0.1 0.2 0.25 0.3 0.35 0.4 0.45 0.5 ; do
  python3 main_generate_parameter_variations.py -c ${BASE_CFG} -N ${NUM_CFG} -s ${STD}

  for N in $(seq 0 $(expr ${NUM_CFG} - 1)) ; do
    NF=$(printf "%03d" ${N})
    CFG="${BASE_CFG}_std${STD}_${NF}"

    if ! [ -f "cfg/${CFG}.json" ] ; then
      echo "cannot find config: ${CFG}"
      exit 1
    fi

    python3 main_recall.py -C ${CSPACE} -c $CFG
  done
done
