# Face detection and recognition using dlib

## Introduction

The main purpose of this repository is to provide an end to end solution for face detection and recognition using the amazing [dlib](http://dlib.net/) library.

## Installation

### Dependencies

The only external dependency of this project is OpenCV >= 3, used only for camera acquistion (or video input).
However, having CUDA and CUDNN will make running the deep learning models for detection more usable.
Dlib will be fetched directly by CMake, so once OpenCV has been installed, the following command should do all the work for you:
 
 ``` bash
 ./build.sh
 ```
 
 Which will download the needed pretrained models and build the project.
 The resulting binary will be found inside of the a newly created directory, named `build`.
 
 ## Running
 
 The program works in the following way:
 
 1. Checks for an enrollments directory with the following structure `enrollments/{person1,person2}` and images of each person inside those directories.
 2. Builds a dictionary of face descriptors of all faces
 3. Opens the Webcam (or video file) and, for each frame, detects, computes face descriptors and compares to dictionary for a match.
 
 All options are customizable, which can be checked by passing the `--help` flag:
 
 ```
 ./build/face --help
Usage: ./build/main [options]
Options:
  --enroll-dir <arg>       Enrollment directory (default: enrollment) 
  --fps <arg>              Force the frames per second for the webcam 
  --help                   Display this help message. 
  --input <arg>            Path to video file to process (defaults to webcam) 
  --light                  Use a lighter detection model 
  --mirror                 Mirror mode (left-right flip) 
  --pyramid-levels <arg>   Pyramid levels for the face detector (default: 1) 
  --scale-factor <arg>     Scaling factor for the input image (default: 1.0) 
  --threshold <arg>        Face recognition threshold (default: 0.5)
```
