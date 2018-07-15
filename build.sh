#!/usr/bin/env bash

# check the models exist
[ -d models ] || mkdir models
# 68 point landmarks model
if [ ! -f models/shape_predictor_68_face_landmarks.dat ]
then
  cd models
  curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
  cd -
fi
# dnn face detector model
if [ ! -f models/mmod_human_face_detector.dat ]
then
  cd models
  curl -O http://dlib.net/files/mmod_human_face_detector.dat.bz2
  bzip2 -d mmod_human_face_detector.dat.bz2
  cd -
fi

# start building
[ -d build ] || mkdir build
cd build
cmake .. -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DUSE_AVX_INSTRUCTIONS=ON \
  -DUSE_SSE2_INSTRUCTIONS=ON \
  -DUSE_SSE4_INSTRUCTIONS=ON \
  -DDLIB_USE_CUDA=OFF

cmake --build . --config Release
cd -

[[ -f compile_commands.json ]] || ln -s build/compile_commands.json .
