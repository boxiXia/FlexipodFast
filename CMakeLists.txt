message(STATUS "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}") # print cmake version
cmake_minimum_required(VERSION 3.14)
project(Titan LANGUAGES C CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF) ## on g++ this ensures: -std=c++11 and not -std=gnu++11

# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17) # set cuda cmake standard to c++17
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_STANDARD 17) # set C++ standard to C++17


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

######################################################################
# use CUDA
find_package(CUDA REQUIRED) # find and include CUDA

###### here you must modify to fit your GPU architecture #####
# check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CMAKE_CUDA_ARCHITECTURES 75)
message(STATUS "CMAKE_CUDA_ARCHITECTURES=" ${CMAKE_CUDA_ARCHITECTURES})
# string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
string(APPEND CMAKE_CUDA_FLAGS " --default-stream per-thread --use_fast_math ") # CUDA_API_PER_THREAD_DEFAULT_STREAM
# https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
##############################################################
#CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
#string(APPEND CMAKE_CUDA_FLAGS " ${ARCH_FLAGS}")
 
#################  use OpenMP ###################################################
# for openmp example,ref: https://bisqwit.iki.fi/story/howto/openmp/
# for simd support,ref: https://nanxiao.gitbooks.io/cuda-little-book/content/posts/use-cmake-to-compile-cuda-program.html
# set(OMP_FLAG_MSVC "-openmp:experimental" "-openmp")
find_package(OpenMP REQUIRED)
IF (OPENMP_FOUND)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-openmp:experimental")
    else()
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    endif()
    #SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
ENDIF()

#SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-std:c++latest")

message(STATUS "OpenMP_CXX_FLAGS=" ${OpenMP_CXX_FLAGS})
message(STATUS ${CMAKE_CUDA_FLAGS})
################################################################################

# find all opengl packages
find_package(glm CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
#find_package(glad CONFIG REQUIRED)
# set ALL_GL_LIBS as a placeholder for all opengl library
#set(ALL_GL_LIBS glm::glm GLEW::GLEW glfw glad::glad)

find_package(msgpack CONFIG REQUIRED)

#find_package(asio CONFIG REQUIRED)

# include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

add_definitions(-DGRAPHICS) # enable this definition to display graphics
#add_definitions(-DVERLET) # enable this definition to integrate via Verlet integration
add_definitions(-DROTATION) # enable this to support rotation in dynamics update
#add_definitions(-DDEBUG_ENERGY) # enable this to debug energy
#add_subdirectory(src/Titan)
add_definitions(-DSTRESS_TEST) # enable stress testing


# macro to create a symbolic link
macro(makeLink src dest target)
    message(STATUS "${target}: link \"${src}\" to \"${dest}\"")
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest}  DEPENDS  ${dest} COMMENT "mklink ${src} -> ${dest}")
endmacro()

add_executable(flexipod 
    src/main.cu 
    src/vec.h src/vec.cu 
    src/shader.h src/shader.cpp 
    src/object.h src/object.cu
    src/sim.h src/sim.cu src/sim.cpp
    src/comonUtils.h src/comonUtils.cpp) 

option(USE_UDP "Enter UDP mode" ON)
if(USE_UDP)
    message(STATUS "UDP ON")
    add_definitions(-DUDP) # enable this definition to send info via DUP
    #target_link_libraries(flexipod PRIVATE asio asio::asio)
    target_sources(flexipod PRIVATE src/network.h src/network.cpp)
endif()

set_target_properties(flexipod PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON # set cuda related properties
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} # set target cuda architecture
                      )

target_include_directories(flexipod PUBLIC ${CUDA_INCLUDE_DIRS} src)

target_link_libraries(flexipod PUBLIC 
        OpenMP::OpenMP_CXX
       # ${ALL_GL_LIBS}
       # glm::glm GLEW::GLEW glfw glad::glad 
       glm::glm GLEW::GLEW glfw
        cuda)# cudart

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:50000000")

# create symbolic link for *.glsl and *.msgpack (this may require administrator privileges)
file(GLOB msgpack_list CONFIGURE_DEPENDS 
    ${CMAKE_SOURCE_DIR}/src/*.msgpack 
    ${CMAKE_SOURCE_DIR}/src/*.glsl)
foreach(data ${msgpack_list})
    get_filename_component(data_name "${data}" NAME)
    makeLink(${data} ${CMAKE_CURRENT_BINARY_DIR}/${data_name} flexipod)
endforeach()



add_executable(testNetwork
    "src/testNetwork.cu"
    src/network.h
    src/network.cpp
)

set_property(TARGET testNetwork PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_link_libraries(testNetwork PRIVATE msgpackc-cxx)
#target_link_libraries(testNetwork PRIVATE asio asio::asio)
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



#find_package(simdjson CONFIG REQUIRED)
#find_package(nlohmann_json CONFIG REQUIRED)

add_executable(jasonTest
    "src/testjson.cu"    
    "src/vec.h" "src/vec.cu"
    "src/testjsonHeader.h" "src/testjsonHeader.cpp"
 "src/comonUtils.h" "src/comonUtils.cpp")
#target_link_libraries(jasonTest PRIVATE simdjson::simdjson simdjson::simdjson-flags simdjson::simdjson-headers)
#target_link_libraries(jasonTest PUBLIC nlohmann_json nlohmann_json::nlohmann_json)
target_link_libraries(jasonTest PRIVATE msgpackc-cxx)
#set_property(TARGET jasonTest PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_link_libraries(jasonTest PUBLIC cuda)
target_include_directories(jasonTest PUBLIC ${CUDA_INCLUDE_DIRS} src)
target_link_libraries(jasonTest PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(jasonTest PROPERTIES 
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
                      )
