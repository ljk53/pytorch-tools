## Quick Start

### Install valgrind
```bash
# On CentOS:
sudo dnf install valgrind valgrind-devel
```

### Run
```bash
./callgrind.sh
```

### Load result
```bash
callgrind_annotate callgrind.out.txt
```
