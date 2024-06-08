cmake_minimum_required(VERSION 3.9)
project(2D_Linear_Programmer C)

set(CMAKE_C_STANDARD 99)

add_executable(2D_Linear_Programmer main.c models.h models.c input_output.h input_output.c floating_number_helper.h floating_number_helper.c)