set -e
set -v

version=0.1.4

git fetch upstream
git merge upstream/develop

export PYTHONROOT=/usr

function pack(){
mkdir -p bin_package
cd bin_package
WITHAVX=$1
WITHMKL=$2
if [ $WITHAVX = "ON" -a $WITHMKL = "OFF" ]; then
    mkdir -p serving-cpu-avx-openblas-$version
    cp ../build_server/output/demo/serving/bin/serving  serving-cpu-avx-openblas-$version
    tar -czvf serving-cpu-avx-openblas-$version.tar.gz serving-cpu-avx-openblas-$version/
fi
if [ $WITHAVX = "OFF" -a $WITHMKL = "OFF" ]; then
    mkdir -p serving-cpu-noavx-openblas-$version
    cp ../build_server/output/demo/serving/bin/serving serving-cpu-noavx-openblas-$version
    tar -czvf serving-cpu-noavx-openblas-$version.tar.gz serving-cpu-noavx-openblas-$version/
fi
if [ $WITHAVX = "ON" -a $WITHMKL = "ON" ]; then
    mkdir -p serving-cpu-avx-mkl-$version
    cp ../build_server/output/demo/serving/bin/* serving-cpu-avx-mkl-$version
    tar -czvf serving-cpu-avx-mkl-$version.tar.gz serving-cpu-avx-mkl-$version/
fi
cd ..
}

function pack_gpu(){
mkdir -p bin_package
cd bin_package
mkdir -p serving-gpu-$version
cp ../build_gpu_server/output/demo/serving/bin/* serving-gpu-$version
cp ../build_gpu_server/third_party/install/Paddle//third_party/install/mklml/lib/* serving-gpu-$version
cp ../build_gpu_server/third_party/install/Paddle//third_party/install/mkldnn/lib/libmkldnn.so.0 serving-gpu-$version
tar -czvf serving-gpu-$version.tar.gz serving-gpu-$version/
cd ..
}

function compile_cpu(){
mkdir -p build_server
cd build_server
WITHAVX=$1
WITHMKL=$2
cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python2.7/ -DPYTHON_LIBRARY=$PYTHONROOT/lib64/libpython2.7.so -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python2.7 -DWITH_AVX=$WITHAVX -DWITH_MKL=$WITHMKL -DSEVER=ON .. > compile_log
make -j20 >> compile_log
#make install >> compile_log
cd ..
pack $WITHAVX $WITHMKL
}

function compile_gpu(){
mkdir -p build_gpu_server
cd build_gpu_server
cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python2.7/ -DPYTHON_LIBRARY=$PYTHONROOT/lib64/libpython2.7.so -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python2.7 -DWITH_GPU=ON -DSERVER=ON .. > compile_log
make -j20 >> compile_log
make install >> compile_log
cd ..
pack_gpu
}

function compile_client(){
mkdir -p build_client
cd build_client
cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python2.7/ -DPYTHON_LIBRARY=$PYTHONROOT/lib64/libpython2.7.so -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python2.7 -DCLIENT=ON -DPACK=ON .. > compile_log
make -j20 >> compile_log
#make install >> compile_log
cd ..
}

function compile_app(){
mkdir -p build_app
cd build_app
cmake -DPYTHON_INCLUDE_DIR=$PYTHONROOT/include/python2.7/ -DPYTHON_LIBRARY=$PYTHONROOT/lib64/libpython2.7.so -DPYTHON_EXECUTABLE=$PYTHONROOT/bin/python2.7 -DAPP=ON ..> compile_log
make -j20 >> compile_log
#make install >> compile_log
cd ..
}
function upload_bin(){
    cd bin_package
    python ../bos_conf/upload.py serving-cpu-avx-openblas-$version.tar.gz
    python ../bos_conf/upload.py serving-cpu-avx-mkl-$version.tar.gz
    python ../bos_conf/upload.py serving-cpu-noavx-openblas-$version.tar.gz
    python ../bos_conf/upload.py serving-gpu-$version.tar.gz
    cd ..
}

#cpu-avx-openblas $1-avx  $2-mkl
#compile_cpu ON OFF

#cpu-avx-mkl
#compile_cpu ON ON

#cpu-noavx-openblas
#compile_cpu OFF OFF

#gpu
#compile_gpu

#client
compile_client

#app
compile_app

#upload bin
#upload_bin
