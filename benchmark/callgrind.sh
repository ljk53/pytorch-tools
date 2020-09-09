#!/bin/bash

set -ue -o pipefail

ROOT="$( cd "$(dirname "$0")"; pwd -P)"

cd $ROOT

use_prebuilt_libtorch() {
  make callgrind
  BIN=./callgrind
}

use_local_libtorch() {
  $ROOT/build_pytorch.sh

  LIBTORCH=local make callgrind
  BIN=./callgrind
}

# On some platforms the ad-hoc makefile doesn't always work, where we can try this one...
use_local_libtorch_cmake() {
  $ROOT/build_pytorch.sh
  BUILD_ROOT=$ROOT/build

  pushd $ROOT
  rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT && cd $BUILD_ROOT
  cmake .. -DCMAKE_PREFIX_PATH=$ROOT/pytorch/torch -DCMAKE_BUILD_TYPE=RelWithDebInfo
  make VERBOSE=1
  popd
  BIN=$BUILD_ROOT/callgrind
}

use_local_libtorch

valgrind \
  --tool=callgrind \
  --callgrind-out-file=callgrind.out.txt \
  --dump-line=yes \
  --instr-atstart=no \
  $BIN

callgrind_annotate \
  --auto=yes \
  --inclusive=yes \
  --tree=both \
  --show-percs=yes \
  --context=16 \
  --include=pytorch \
  callgrind.out.txt
