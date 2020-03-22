cmake_minimum_required(VERSION 3.14)
project(Titan LANGUAGES C CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)
set(CMAKE_CXX_STANDARD 17) # set C++ standard to C++17

# SET CMAKE_TOOLCHAIN_FILE
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

# set cmake module path (where .cmake files are stored) (not used yet)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")





# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# use OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# use CUDA
find_package(CUDA REQUIRED) # find and include CUDA

###### here you must modify to fit your GPU architecture #####
# check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
##############################################################
#CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
#string(APPEND CMAKE_CUDA_FLAGS " ${ARCH_FLAGS}")
message(STATUS ${CMAKE_CUDA_FLAGS})
  
  
# find all opengl packages
find_package(glm CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
# set ALL_GL_LIBS as a placeholder for all opengl library
set(ALL_GL_LIBS GLEW::GLEW glm glfw glad::glad)

find_package(msgpack CONFIG CONFIG)

# include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

add_definitions(-DGRAPHICS)
add_definitions(-DCONSTRAINTS)

#add_subdirectory(src/Titan)

add_executable(main src/main.cu src/vec.h src/vec.cu)

set_target_properties(main PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main PUBLIC ${CUDA_INCLUDE_DIRS} src)

target_link_libraries(main PRIVATE 
        OpenMP::OpenMP_CXX
        ${ALL_GL_LIBS})

#target_link_libraries(main PRIVATE cuda) # cudart


#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:50000000")

