all : microbench

libtorch-shared-with-deps-1.6.0%2Bcpu.zip :
	curl -LsO 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip'

linux/libtorch : libtorch-shared-with-deps-1.6.0%2Bcpu.zip
	unzip -qq -o $< -d linux

libtorch : linux/libtorch
	ln -s $< $@

microbench : microbench.cpp libtorch
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
