#!/bin/bash

set -ue -o pipefail

ROOT="$( cd "$(dirname "$0")"; pwd -P)"

cd $ROOT

make

valgrind --tool=callgrind --callgrind-out-file=callgrind.out.txt ./callgrind --instr-atstart=no

callgrind_annotate callgrind.out.txt
