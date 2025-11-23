cmake_minimum_required(VERSION 3.10)
project(MTE301_AMR_Delivery_Robot CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(amr_sim
    src/main.cpp
)
