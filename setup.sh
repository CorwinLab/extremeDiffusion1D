#it's somewhat important that you run this script from the pyCudaPacking
#directory

pyCudaPath=${PWD}

#first we need to ask where you want to put pybind11
while true; do
    read -rep $'Please input the path where you would like pybind11. Leave blank to put in the same directory as pyCudaPacking.\n' pyBindPath
    if [ "$pyBindPath" = "" ]; then
	pyBindPath=$(echo $pyCudaPath | sed "s|/pyCudaPacking||")
	echo $pyBindPath
    fi
    break
done

#then np_quad:
while true; do
    read -rep $'Please input the path where you would like np_quad. Leave blank to put in the same directory as pyCudaPacking.\n' npQuadPath
    if [ "$npQuadPath" = "" ]; then
	npQuadPath=$(echo $pyCudaPath | sed "s|/pyCudaPacking||")
	echo $npQuadPath
    fi
    break
done

cd $pyBindPath
# clone the standard version of pybind11
git clone https://github.com/pybind/pybind11.git

cd $npQuadPath
# clone the standard version of np_quad
git clone git@github.com:SimonsGlass/numpy_quad.git
cd numpy_quad/
echo "Now setup your numpy_quad"
python3 setup.py develop --user

cd ../src/
c++ -O3 -march=native -Wall -shared -std=gnu++11 -fPIC $(python3-config --includes) libDiffusion.cpp -I/c/modular-boost -lquadmath -o libDiffusion.so -I"$pyBindPath/pybind11/include"