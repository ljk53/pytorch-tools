LIBTORCH ?= linux
CC       ?= clang
CXX      ?= clang++

all : microbench

macos/libtorch :
	curl -LsO https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip
	unzip -qq -o libtorch-macos-1.6.0.zip -d macos

linux/libtorch :
	curl -LsO 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip'
	unzip -qq -o libtorch-shared-with-deps-1.6.0%2Bcpu.zip -d linux

local/libtorch :
	mkdir local
	ln -s ../pytorch/torch local/libtorch

microbench : microbench.cpp $(LIBTORCH)/libtorch
	$(CXX) -std=c++14 -O3 \
	-I .. \
	-I $(LIBTORCH)/libtorch/include \
	-I $(LIBTORCH)/libtorch/include/torch/csrc/api/include \
	-L $(LIBTORCH)/libtorch/lib \
	microbench.cpp \
	-o $@ \
	-Wl,-rpath,$(LIBTORCH)/libtorch/lib \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm -rf microbench linux libtorch*
