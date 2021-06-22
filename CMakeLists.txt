message(STATUS "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}") # print cmake version
cmake_minimum_required(VERSION 3.14)
project(SoftSim LANGUAGES C CXX CUDA) # use CXX, CUDA by default (since CUDA is a language, don't need cuda_add_executable)

# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
set(CMAKE_CUDA_STANDARD 17) # set cuda cmake standard to c++17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # cuda separable compilation
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # position independent code

set(CMAKE_CXX_STANDARD 17) # set c++ standard to 17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # https://cmake.org/cmake/help/git-stage/prop_tgt/CXX_EXTENSIONS.html
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # If enabled, generates a compile_commands.json file of compiler calls
if(MSVC)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-std:c++${CMAKE_CXX_STANDARD}") # -Xcompiler=-std:c++latest"
endif()

message(STATUS "CMAKE_CUDA_STANDARD=" ${CMAKE_CUDA_STANDARD})
message(STATUS "CMAKE_CXX_STANDARD=" ${CMAKE_CXX_STANDARD})


# https://github.com/microsoft/vcpkg/blob/master/docs/users/integration.md#using-an-environment-variable-instead-of-a-command-line-option
# SET CMAKE_TOOLCHAIN_FILE
# you can add VCPKG_ROOT= your/vcpkg/root/folder as environment variable
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

# set cmake module path (where .cmake files are stored)
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
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/openmp:experimental")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
    else()
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    endif()
    #SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 
ENDIF()


message(STATUS "OpenMP_CXX_FLAGS=" ${OpenMP_CXX_FLAGS})
message(STATUS "CMAKE_CUDA_FLAGS=" ${CMAKE_CUDA_FLAGS})
################################################################################

# find all opengl packages
find_package(glm CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
#find_package(glad CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED) # GUI
find_package(msgpack CONFIG REQUIRED)
find_package(clipp CONFIG REQUIRED)
#find_package(asio CONFIG REQUIRED)
#find_package(ghc_filesystem CONFIG REQUIRED)

# include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

add_definitions(-DGRAPHICS) # enable this definition to display graphics
#add_definitions(-DVERLET) # enable this definition to integrate via Verlet integration
add_definitions(-DROTATION) # enable this to support rotation in dynamics update
#add_definitions(-DDEBUG_ENERGY) # enable this to debug energy
#add_subdirectory(src/Titan)
add_definitions(-DSTRESS_TEST) # enable stress testing (spring info)
add_definitions(-DMEASURE_CONSTRAINT) # enable constraint measurement



# macro to create a symbolic link
macro(makeLink src dest target)
    message(STATUS "${target}: link \"${src}\" to \"${dest}\"")
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest}  DEPENDS  ${dest} COMMENT "mklink ${src} -> ${dest}")
endmacro()

add_executable(flexipod 
    src/main.cpp 
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

target_include_directories(flexipod PUBLIC ${CUDA_INCLUDE_DIRS} src)
target_link_libraries(flexipod PUBLIC 
    OpenMP::OpenMP_CXX
    glm::glm GLEW::GLEW glfw imgui::imgui # glad::glad 
    cuda # cudart
    clipp::clipp
    #ghcFilesystem::ghc_filesystem
)

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:50000000")

# create symbolic link for *.glsl and *.msgpack (this may require administrator privileges)
file(GLOB msgpack_list CONFIGURE_DEPENDS 
    ${CMAKE_SOURCE_DIR}/src/*.msgpack 
    ${CMAKE_SOURCE_DIR}/src/shader
    ${CMAKE_SOURCE_DIR}/src/fonts/*.ttf
)
foreach(data ${msgpack_list})
    get_filename_component(data_name "${data}" NAME)
    makeLink(${data} ${CMAKE_CURRENT_BINARY_DIR}/${data_name} flexipod)
endforeach()


add_executable(testNetwork
    "src/testNetwork.cu"
    src/network.h
    src/network.cpp
)
target_link_libraries(testNetwork PRIVATE msgpackc-cxx)
#target_link_libraries(testNetwork PRIVATE asio asio::asio)
target_link_libraries(testNetwork PRIVATE cuda)
target_include_directories(testNetwork PUBLIC ${CUDA_INCLUDE_DIRS} src)



add_executable(mathTest
    src/mathTest.cu
    src/vec.h src/vec.cu
)
target_link_libraries(mathTest PRIVATE msgpackc-cxx)
target_link_libraries(mathTest PRIVATE cuda)
target_link_libraries(mathTest PRIVATE clipp::clipp)
#target_link_libraries(mathTest PRIVATE ghcFilesystem::ghc_filesystem)
target_include_directories(mathTest PUBLIC ${CUDA_INCLUDE_DIRS} src)


add_executable(jasonTest
    "src/testjson.cu"    
    "src/vec.h" "src/vec.cu"
    "src/testjsonHeader.h" "src/testjsonHeader.cpp"
 "src/comonUtils.h" "src/comonUtils.cpp")
target_include_directories(jasonTest PUBLIC ${CUDA_INCLUDE_DIRS} src)
target_link_libraries(jasonTest PRIVATE msgpackc-cxx)
target_link_libraries(jasonTest PUBLIC cuda)
target_link_libraries(jasonTest PUBLIC OpenMP::OpenMP_CXX)


# set properties for all targets
set_target_properties(flexipod mathTest jasonTest testNetwork
    PROPERTIES
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} # set target cuda architecture
    POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE}    
    CUDA_SEPARABLE_COMPILATION ${CMAKE_CUDA_SEPARABLE_COMPILATION}# separable compilation
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED}
    CXX_EXTENSIONS ${CMAKE_CXX_EXTENSIONS}
)