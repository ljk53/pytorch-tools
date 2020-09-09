#!/bin/bash

set -ue -o pipefail

ROOT="$( cd "$(dirname "$0")"; pwd -P)"
PYTORCH_ROOT="${PYTORCH_ROOT:-$ROOT/pytorch}"

use_prebuilt_libtorch() {
  cd $ROOT
  make callgrind
  BIN=$ROOT/callgrind
}

use_local_libtorch() {
  cd $ROOT
  LIBTORCH=local make callgrind
  BIN=$ROOT/callgrind
}

# On some platforms the ad-hoc makefile doesn't always work, where we can try this one...
use_local_libtorch_cmake() {
  BUILD_ROOT=$ROOT/build

  rm -rf $BUILD_ROOT && mkdir -p $BUILD_ROOT
  pushd $BUILD_ROOT
  cmake .. -DCMAKE_PREFIX_PATH=$PYTORCH_ROOT/torch -DCMAKE_BUILD_TYPE=RelWithDebInfo
  make VERBOSE=1
  popd
  BIN=$BUILD_ROOT/callgrind
}

if [ "${LIBTORCH:-}" == "local" ]; then
  $ROOT/build_pytorch.sh
  use_local_libtorch
else
  use_prebuilt_libtorch
fi

valgrind \
  --tool=callgrind \
  --callgrind-out-file=callgrind.out.txt \
  --dump-line=yes \
  --dump-instr=yes \
  --collect-jumps=yes \
  --instr-atstart=no \
  $BIN

callgrind_annotate \
  --auto=yes \
  --inclusive=yes \
  --tree=both \
  --show-percs=yes \
  --context=16 \
  --include=$PYTORCH_ROOT \
  callgrind.out.txt
