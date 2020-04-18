#!/bin/sh

CFG=final_config

ROOT=$(dirname $0)

if [ ! -d $ROOT/out/recall/$CFG/ ]; then
  echo "need to create RECALL data first"
  exit 1
fi

echo -n "RECALL "

python3 $ROOT/out/recall/parse_results.py $ROOT/out/recall/final_config/cspace*

echo

if [ ! -d $ROOT/out/copy/$CFG/ ]; then
  echo "need to create COPY data first"
  exit 1
fi

echo -n "COPY "

python3 $ROOT/out/copy/parse_results.py $ROOT/out/copy/final_config/cspace*

echo

if [ ! -f $ROOT/out/compare/$CFG/cspace0/readout_*.pdf ]; then
  echo "need to create COMPARE data first"
  exit 1
fi

echo "COMPARE:"

for D in $ROOT/out/compare/$CFG/cspace* ; do
  CSPACE=$(basename $D)
  F=$(echo $D/readout_*.pdf)
  echo "  $CSPACE: see generated figure $F"
done

echo

if [ ! -f $ROOT/out/frankland_greene/experiment_1/$CFG/cspace0/results.json ]; then
  echo "need to create Frankland & Greene experiment 1 data first"
  exit 1
fi

echo "Frankland & Greene experiment 1 results:"

for D in $ROOT/out/frankland_greene/experiment_1/$CFG/cspace* ; do
  CSPACE=$(basename $D)
  EC=$(grep "\"error_C\": " $D/results.json | sed 's/.*: //; s/,$//')
  EUV=$(grep "\"error_UV\": " $D/results.json | sed 's/.*: //; s/,$//')
  echo "  $CSPACE:  error C: $EC  error S(agent) + S(patient): $EUV"
done

echo

if [ ! -f $ROOT/out/frankland_greene/experiment_2/$CFG/cspace0/results.json ]; then
  echo "need to create Frankland & Greene experiment 2 data first"
  exit 1
fi

echo "Frankland & Greene experiment 2 results:"

for D in $ROOT/out/frankland_greene/experiment_2/$CFG/cspace* ; do
  CSPACE=$(basename $D)
  EV=$(grep "\"error_V\": " $D/results.json | sed 's/.*: //; s/,$//')
  EU=$(grep "\"error_U\": " $D/results.json | sed 's/.*: //; s/,$//')
  echo "  $CSPACE:  error S(agent): $EV  error S(patient): $EU"
done
