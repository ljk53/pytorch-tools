LIBTORCH ?= linux

all : microbench

linux/libtorch :
	curl -LsO 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip'
	unzip -qq -o libtorch-shared-with-deps-1.6.0%2Bcpu.zip -d linux

libtorch.linux : linux/libtorch
	@echo "Use downloaded libtorch"
	rm -f libtorch
	ln -s $< libtorch

libtorch.local : ../pytorch/torch
	@echo "Use locally built PyTorch"
	rm -f libtorch
	ln -s $< libtorch

microbench : microbench.cpp libtorch.$(LIBTORCH)
	g++ -std=c++14 -O3 \
	-I .. \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	microbench.cpp \
	-o $@ \
	-Wl,-rpath,libtorch/lib \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm -rf microbench linux libtorch*
