#!/bin/sh

set -xe

# build module, install by default

modname="vb_module"
now=$(date "+%F-%H-%M")

cd $(dirname $0)
code_dir=$(pwd)/src

mkdir build 2>/dev/null
cd build
echo "in directory $(pwd)"

# create base-directory
base="nest-${modname}-${now}"
echo "create base directory: $base"
mkdir $base
cd $base

# copy code directory
code=$(basename ${code_dir})
echo "copy code directory: $code"
cp -R ${code_dir} .

# create build directory
build="build"
echo "create build directory: $build"
mkdir $build
cd $build

# configure
pwd
echo "run cmake..."
echo $NEST_INSTALL_DIR
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config -Dwith-python=3 ../${code}

if [ $? -ne 0 ]; then
  echo "cmake failed, aborting."
  exit -1
fi

echo "cmake successful."

# make
echo "make..."
make -j 8

if [ $? -ne 0 ]; then
  echo "make failed, aborting."
  exit -1
fi

echo "make successful."

# install
echo "installing..."
make install

if [ $? -ne 0 ]; then
  echo "make install failed, aborting."
  exit -1
fi

echo "make install successful."
