#!/bin/bash

set -uf -o pipefail

SRC_ROOT=$HOME/src/pytorch
WORK_DIR=$(mktemp -d -t size-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX)

# Set path to the bloaty executable (https://github.com/google/bloaty)
BLOATY_BIN=${BLOATY_BIN:-bloaty}
STRIP_BIN=${STRIP_BIN:-$ANDROID_NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-strip}

INPUT=input.csv

KEYWORDS=(
  "torch::CppFunction::makeUnboxedOnly"
  "torch::CppFunction::"

  "c10::Dispatcher::call"
  "c10::Dispatcher::"

  "c10::KernelFunction::makeFromUnboxedOnlyRuntimeFunction"
  "c10::KernelFunction::makeFromUnboxedRuntimeFunction"
  "c10::KernelFunction::make_boxed_function"
  "c10::KernelFunction::call"
  "c10::KernelFunction::"

  "c10::impl::BoxedKernelWrapper"
  "c10::impl::make_boxed_from_unboxed_functor"
  "c10::impl::wrap_kernel_functor_unboxed_"
  "c10::impl::detail::WrapFunctionIntoRuntimeFunctor_"
  "c10::impl::detail::WrapFunctionIntoFunctor_"
  "c10::impl::OperatorEntry::assertSignatureIsCorrect"
  "c10::impl::"

  "::callUnboxedKernel"

  "src/ATen/Functions.cpp"
)

build_android() {
  pushd $SRC_ROOT

  scripts/build_android.sh \
    -DUSE_STATIC_DISPATCH=OFF \
    -DBUILD_BINARY=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DANDROID_DEBUG_SYMBOLS=ON

  cp build_android/install/bin/speed_benchmark_torch $WORK_DIR/target.debug

  $STRIP_BIN $WORK_DIR/target.debug -o $WORK_DIR/target.strip

  popd
}

run_bloaty() {
  $BLOATY_BIN \
    --demangle=full \
    --csv -n 0 -d sections,compileunits,symbols \
    --debug-file=$WORK_DIR/target.debug \
    $WORK_DIR/target.strip > $INPUT
}

analyze() {
  CUR=0
  cp $INPUT "$WORK_DIR/s.$CUR"

  for K in ${KEYWORDS[*]}; do
    NEXT=$((CUR+1))
    CUR_FILE="$WORK_DIR/s.$CUR"
    NEXT_FILE="$WORK_DIR/s.$NEXT"
    RESULT_FILE="$WORK_DIR/r.${K//\//-}"

    # filter by the current keyword
    cat $CUR_FILE | grep $K > $RESULT_FILE
    cat $CUR_FILE | grep -v $K > $NEXT_FILE
    CUR=$NEXT

    # calculate total size
    SIZE=$(cat $RESULT_FILE | awk 'BEGIN {FS=","} {s+=$NF} END {print s}')
    COUNT=$(wc -l $RESULT_FILE| cut -d' ' -f-1)
    printf "%-60s\t%8d\t%5d\n" $K $SIZE $COUNT
  done | sort -rnk2

  echo
  echo DONE!
  echo Check intermediate result: $WORK_DIR
}

build_android
run_bloaty
analyze
