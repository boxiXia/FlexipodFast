message(STATUS "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}") # print cmake version
cmake_minimum_required(VERSION 3.14)
project(Titan LANGUAGES C CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)
set(CMAKE_CXX_STANDARD 20) # set C++ standard to C++20

# https://github.com/microsoft/vcpkg/blob/master/docs/users/integration.md#using-an-environment-variable-instead-of-a-command-line-option
# SET CMAKE_TOOLCHAIN_FILE
# you can add VCPKG_ROOT= your/vcpkg/root/folder as environment variable
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

# set cmake module path (where .cmake files are stored) (not used yet)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

#set(CMAKE_BUILD_TYPE Release)
#list(APPEND CMAKE_CXX_FLAGS " -O2 ")

# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# for simd support,ref: https://gitlab.kitware.com/cmake/cmake/-/merge_requests/3301
set(OMP_FLAG_MSVC "-openmp:experimental" "-openmp")

# use OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# use CUDA
find_package(CUDA REQUIRED) # find and include CUDA



###### here you must modify to fit your GPU architecture #####
# check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CMAKE_CUDA_ARCHITECTURES=" ${CMAKE_CUDA_ARCHITECTURES})
# string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
string(APPEND CMAKE_CUDA_FLAGS " --default-stream per-thread")
# https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
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

find_package(asio CONFIG REQUIRED)

# include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

add_definitions(-DGRAPHICS) # enable this definition to display graphics
#add_definitions(-DVERLET) # enable this definition to integrate via Verlet integration
add_definitions(-DROTATION) # enable this to support rotation in dynamics update
#add_definitions(-DDEBUG_ENERGY) # enable this to debug energy
#add_subdirectory(src/Titan)




add_executable(flexipod 
    src/main.cu 
    src/vec.h src/vec.cu 
    src/shader.h src/shader.cpp 
    src/object.h src/object.cu
    src/sim.h src/sim.cu
    ) 
# set target cuda architecture
set_property(TARGET flexipod PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})


option(USE_UDP "Enter UDP mode" ON)
if(USE_UDP)
    message(STATUS "UDP ON")
    add_definitions(-DUDP) # enable this definition to send info via DUP
    target_link_libraries(flexipod PRIVATE asio asio::asio)
    target_sources(flexipod PRIVATE src/network.h src/network.cpp)
endif()

set_target_properties(flexipod PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(flexipod PUBLIC ${CUDA_INCLUDE_DIRS} src)

target_link_libraries(flexipod PRIVATE 
        OpenMP::OpenMP_CXX
        ${ALL_GL_LIBS}
        cuda)# cudart

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:50000000")

# copy shaders to binary directory
add_custom_command(
        TARGET flexipod POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/src/shaderVertex.glsl
                ${CMAKE_CURRENT_BINARY_DIR}/shaderVertex.glsl)
add_custom_command(
        TARGET flexipod POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/src/shaderFragment.glsl
                ${CMAKE_CURRENT_BINARY_DIR}/shaderFragment.glsl)

add_executable(testNetwork
    "src/testNetwork.cu"
    src/network.h
    src/network.cpp
)
set_property(TARGET testNetwork PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_link_libraries(testNetwork PRIVATE msgpackc-cxx)
target_link_libraries(testNetwork PRIVATE asio asio::asio)
target_link_libraries(testNetwork PRIVATE cuda)
target_include_directories(testNetwork PUBLIC ${CUDA_INCLUDE_DIRS} src)



add_executable(mathTest
    src/mathTest.cu
    src/vec.h src/vec.cu
)
set_property(TARGET mathTest PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_link_libraries(mathTest PRIVATE msgpackc-cxx)
target_link_libraries(mathTest PRIVATE cuda)
target_include_directories(mathTest PUBLIC ${CUDA_INCLUDE_DIRS} src)

add_executable(
    testMsgpack
    "src/msgpackTest.cu"
    src/vec.h src/vec.cu)
set_property(TARGET testMsgpack PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_link_libraries(testMsgpack PRIVATE msgpackc-cxx)
target_link_libraries(testMsgpack PRIVATE cuda)
target_include_directories(testMsgpack PUBLIC ${CUDA_INCLUDE_DIRS} src)



