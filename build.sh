#!/usr/bin/env bash

# Environment
[ ! -z ${DLIB_USE_CUDA} ] || DLIB_USE_CUDA=ON

function find_gcc() {
  [ `which gcc-7 2>/dev/null` ] && echo "gcc-7" && exit
  [ `which gcc-6 2>/dev/null` ] && echo "gcc-6" && exit
  [ `which gcc-5 2>/dev/null` ] && echo "gcc-5" && exit
  [ `which gcc 2>/dev/null` ] && echo "gcc" && exit
}

# check the models exist
[ -d models ] || mkdir models
# 5 point landmarks model
if [ ! -f models/shape_predictor_5_face_landmarks.dat ]
then
  cd models
  curl -O http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
  bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
  cd -
fi
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
# face recognition model
if [ ! -f models/dlib_face_recognition_resnet_model_v1.dat ]
then
  cd models
  curl -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
  bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
  cd -
fi

# start building
[ -d build ] || mkdir build
cd build
CC=`find_gcc` \
cmake .. -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DUSE_AVX_INSTRUCTIONS=ON \
  -DUSE_SSE2_INSTRUCTIONS=ON \
  -DUSE_SSE4_INSTRUCTIONS=ON \
  -DDLIB_USE_CUDA=${DLIB_USE_CUDA}

cmake --build . --config Release
cd -

[[ -f compile_commands.json ]] || ln -s build/compile_commands.json .
