#!/bin/sh

# create 5 new content space instances
NUMCSPACE=5

for i in $(seq 0 $(expr $NUMCSPACE - 1)) ; do
  python3 main_train_content_space.py -C cspace$i
done

# run 50 tests of create/recall (10 for each content assembly) for each
# content space instance

for i in $(seq 0 $(expr $NUMCSPACE - 1)) ; do
  python3 main_recall.py -C cspace$i
done

# run 10 tests of copy/recall (2 for each content assembly) for each
# content space instance

for i in $(seq 0 $(expr $NUMCSPACE - 1)) ; do
  python3 main_copy.py -C cspace$i
done

# run compare (only on first content space instance to save time)

python3 main_compare.py -C cspace0

# run Frankland & Greene experiments (only on first content space instance to
# save time)

python3 main_frankland_and_greene.py -C cspace0 -e 1
python3 main_frankland_and_greene.py -C cspace0 -e 2
