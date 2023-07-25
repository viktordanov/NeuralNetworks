cmake_minimum_required(VERSION 3.25)
project(cpp_neural_networks)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(cpp_neural_networks
        main.cpp
        Matrix.cpp
        Matrix.hpp
        NeuralNetwork.cpp
        NeuralNetwork.hpp
        Util.cpp
        Util.hpp)
