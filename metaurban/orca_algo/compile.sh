mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DFULL_OUTPUT_FLAG=ON -DFULL_LOG_FLAG=ON -DMAPF_LOG_FLAG=OFF ..
make -j8
