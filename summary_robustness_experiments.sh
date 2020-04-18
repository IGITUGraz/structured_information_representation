#!/bin/sh

BASE_CFG=final_config

ROOT=$(dirname $0)

for STD in 0.1 0.2 0.25 0.3 0.35 0.4 0.45 0.5 ; do
  echo "STD: $STD"

  if [ ! -d $ROOT/out/recall/${BASE_CFG}_std${STD}_000 ]; then
    echo "need to create RECALL data first"
    exit 1
  fi

  python3 $ROOT/out/recall/parse_results.py $ROOT/out/recall/${BASE_CFG}_std${STD}_*/cspace*

  echo
done
