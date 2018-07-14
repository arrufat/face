#!/usr/bin/env bash

# check the models exist
[ -d models ] || mkdir models
if [ ! -f models/shape_predictor_68_face_landmarks.dat ]
then
  cd models
  curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
  cd -
fi

# start building
[ -d build ] || mkdir build
cd build
cmake .. -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DUSE_AVX_INSTRUCTIONS=1 \
  -DUSE_SSE2_INSTRUCTIONS=1 \
  -DUSE_SSE4_INSTRUCTIONS=1 \
  -DDLIB_USE_CUDA=0

cmake --build . --config Release
cd -
[[ -f compile_commands.json ]] || ln -s build/compile_commands.json .
