LIBTORCH     ?= linux
CC           ?= clang
CXX          ?= clang++
CXX11_ABI    ?= 0
PYTORCH_ROOT ?= $(CURDIR)/pytorch

CXX_FLAGS     = -std=c++14 -O3 -D_GLIBCXX_USE_CXX11_ABI=$(CXX11_ABI)

INCLUDE_FLAGS = -I $(LIBTORCH)/libtorch/include \
                -I $(LIBTORCH)/libtorch/include/torch/csrc/api/include

LDFLAGS       = -L $(LIBTORCH)/libtorch/lib \
                -Wl,-rpath,$(LIBTORCH)/libtorch/lib \
                -lc10 -ltorch -ltorch_cpu

all : microbench boxing matrix_indexing callgrind

macos/libtorch :
	curl -LsO https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip
	unzip -qq -o libtorch-macos-1.6.0.zip -d macos

linux/libtorch :
	curl -LsO 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip'
	unzip -qq -o libtorch-shared-with-deps-1.6.0%2Bcpu.zip -d linux

rpi/libtorch :
	curl -LsO 'https://github.com/ljk53/pytorch-rpi/raw/master/libtorch-rpi-armv7l-cxx11-abi-shared-without-deps-1.6.0.zip'
	unzip -qq -o libtorch-rpi-armv7l-cxx11-abi-shared-without-deps-1.6.0.zip -d rpi

rpi64/libtorch :
	curl -LsO 'https://github.com/ljk53/pytorch-rpi/raw/master/libtorch-rpi-aarch64-cxx11-abi-shared-without-deps-1.6.0.zip'
	unzip -qq -o libtorch-rpi-aarch64-cxx11-abi-shared-without-deps-1.6.0.zip -d rpi64

local/libtorch :
	mkdir -p local
	ln -s $(PYTORCH_ROOT)/torch local/libtorch

microbench : microbench.cpp common.h $(LIBTORCH)/libtorch
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) microbench.cpp $(LDFLAGS) -o $@

boxing : boxing.cpp common.h $(LIBTORCH)/libtorch
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) boxing.cpp $(LDFLAGS) -o $@

matrix_indexing : matrix_indexing.cpp common.h $(LIBTORCH)/libtorch
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) matrix_indexing.cpp $(LDFLAGS) -o $@

callgrind: callgrind.cpp $(LIBTORCH)/libtorch
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) callgrind.cpp $(LDFLAGS) -o $@

clean:
	rm -rf macos linux rpi rpi64 local libtorch*
	rm -rf build
	rm -rf microbench
	rm -rf boxing
	rm -rf matrix_indexing
	rm -rf callgrind
