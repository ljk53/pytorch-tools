all : mm

libtorch-shared-with-deps-1.6.0%2Bcpu.zip :
	curl -LsO 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip'

linux/libtorch : libtorch-shared-with-deps-1.6.0%2Bcpu.zip
	unzip -qq -o $< -d linux

mm : mm.cpp linux/libtorch
	rm -f libtorch
	ln -s linux/libtorch libtorch
	g++ -std=c++14 \
	-I .. \
	-I libtorch/include \
	-I libtorch/include/torch/csrc/api/include \
	-L libtorch/lib \
	mm.cpp \
	-o $@ \
	-Wl,-rpath,libtorch/lib \
	-lc10 -ltorch -ltorch_cpu \
	-D_GLIBCXX_USE_CXX11_ABI=1

clean:
	rm -rf mm linux libtorch*
