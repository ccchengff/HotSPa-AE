ENV_DIR=$(pwd)/env
mkdir -p $ENV_DIR
cd $ENV_DIR

# 1. prepare Gcc-9.5.0
wget https://ftp.gnu.org/gnu/gcc/gcc-9.5.0/gcc-9.5.0.tar.gz
tar -zxf gcc-9.5.0.tar.gz

mkdir -p gcc-9.5.0
cd gcc-9.5.0
GCC_PATH=$(pwd)
./contrib/download_prerequisites 
./configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=${GCC_PATH} --enable-threads=posix
make -j 32
sudo make install
cd ..

echo "" >> init.sh
echo "export PATH=${GCC_PATH}/bin:\$PATH" >> init.sh
echo "export LD_LIBRARY_PATH=${GCC_PATH}/lib:${GCC_PATH}/lib64:\$LD_LIBRARY_PATH" >> init.sh
source init.sh

# 2. prepare CMake-3.24
sudo apt-get remove cmake cmake-gui
sudo apt-get update
sudo apt-get install wget build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2.tar.gz
tar -zxf cmake-3.24.2.tar.gz

mkdir -p cmake-3.24.2
cd cmake-3.24.2
CMAKE_PATH=$(pwd)
./configure --prefix=${CMAKE_PATH}
make -j 32
sudo make install
cd ..

echo "" >> init.sh
echo "export PATH=${CMAKE_PATH}/bin:\$PATH" >> init.sh
source init.sh