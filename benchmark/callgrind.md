# Quick Start

## Install Valgrind

### On CentOS
```bash
sudo dnf install valgrind valgrind-devel
```

### On Mac
Follow the instructions at: https://stackoverflow.com/questions/58360093/how-to-install-valgrind-on-macos-catalina-10-15-with-homebrew
```bash
brew tap LouisBrunner/valgrind
brew install --HEAD LouisBrunner/valgrind/valgrind
```

## Build & Run

### Use locally built libtorch
This script will checkout pytorch source code and build it from scratch.
You can use this workflow to modify pytorch code locally and run the benchmark against the modified version.
```bash
LIBTORCH=local ./callgrind.sh
```
If you already a working copy of PyTorch repo, you could use it instead:
```bash
PYTORCH_ROOT=<your_pytorch_src_repo> LIBTORCH=local ./callgrind.sh
```

### Use prebuilt libtorch downloaded from official website
```bash
# On Linux
./callgrind.sh

# On Mac
LIBTORCH=macos ./callgrind.sh
```

## Show Annotated Result

Use command line flags: https://www.valgrind.org/docs/manual/cl-manual.html

```bash
callgrind_annotate \
  --auto=yes \
  --inclusive=yes \
  --tree=both \
  --show-percs=yes \
  --context=16 \
  --include=pytorch \
  callgrind.out.txt
```

## Visualize Result

### Install Kcachegrind
```bash
# On CentOS
sudo dnf install kcachegrind

# On Mac
brew install qcachegrind
```
