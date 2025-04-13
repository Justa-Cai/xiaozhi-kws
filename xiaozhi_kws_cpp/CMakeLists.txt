cmake_minimum_required(VERSION 3.10)
project(xiaozhi_kws_cpp VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置rpath，确保可以找到共享库
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# 查找必要的库
find_library(FFTW3_LIB fftw3 REQUIRED)
find_library(FFTW3F_LIB fftw3f REQUIRED)
find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)

# 查找JSON库
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp PATHS 
    /usr/include 
    /usr/local/include
    REQUIRED)

# 查找libsndfile
find_library(SNDFILE_LIB sndfile REQUIRED)
find_path(SNDFILE_INCLUDE_DIR sndfile.h REQUIRED)

# 查找FFmpeg库
find_library(AVFORMAT_LIB avformat REQUIRED)
find_library(AVCODEC_LIB avcodec REQUIRED)
find_library(AVUTIL_LIB avutil REQUIRED)
find_library(SWRESAMPLE_LIB swresample REQUIRED)
find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h REQUIRED)

# 查找本地ONNX Runtime
find_library(ONNXRUNTIME_LIB onnxruntime PATHS
    ${CMAKE_SOURCE_DIR}/lib
    REQUIRED)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS
    ${CMAKE_SOURCE_DIR}/include/onnxruntime
    REQUIRED)

# 添加include目录
include_directories(include)
include_directories(${FFTW3_INCLUDE_DIR})
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
include_directories(${SNDFILE_INCLUDE_DIR})
include_directories(${FFMPEG_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# 打印库信息
message(STATUS "FFTW3 libraries: ${FFTW3_LIB}")
message(STATUS "FFTW3F libraries: ${FFTW3F_LIB}")
message(STATUS "FFTW3 include: ${FFTW3_INCLUDE_DIR}")
message(STATUS "JSON include: ${NLOHMANN_JSON_INCLUDE_DIR}")
message(STATUS "SNDFILE library: ${SNDFILE_LIB}")
message(STATUS "SNDFILE include: ${SNDFILE_INCLUDE_DIR}")
message(STATUS "AVFORMAT library: ${AVFORMAT_LIB}")
message(STATUS "AVCODEC library: ${AVCODEC_LIB}")
message(STATUS "AVUTIL library: ${AVUTIL_LIB}")
message(STATUS "SWRESAMPLE library: ${SWRESAMPLE_LIB}")
message(STATUS "FFMPEG include: ${FFMPEG_INCLUDE_DIR}")
message(STATUS "ONNXRUNTIME library: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNXRUNTIME include: ${ONNXRUNTIME_INCLUDE_DIR}")

# 构建后处理器测试程序
add_executable(test_post_processor 
    src/post_processor.cpp
    tests/test_post_processor.cpp)

# 构建特征提取器测试程序
add_executable(feature_extractor_test
    examples/feature_extractor_test.cpp
    src/feature_extractor.cpp
    src/config_parser.cpp
    src/audio_utils.cpp)

target_link_libraries(feature_extractor_test
    ${FFTW3_LIB}
    ${FFTW3F_LIB}
    ${SNDFILE_LIB}
    ${AVFORMAT_LIB}
    ${AVCODEC_LIB}
    ${AVUTIL_LIB}
    ${SWRESAMPLE_LIB})

# 批量检测器库源文件
set(DETECTOR_LIB_SOURCES
    src/detector.cpp
    src/model.cpp
    src/feature_extractor.cpp
    src/config_parser.cpp
    src/audio_utils.cpp
    src/post_processor.cpp
    src/mfcc_features.cpp
    src/xiaozhi_kws.cpp)

# 构建批量检测器程序
add_executable(batch_detector
    examples/batch_detector.cpp
    ${DETECTOR_LIB_SOURCES})

target_link_libraries(batch_detector
    ${FFTW3_LIB}
    ${FFTW3F_LIB}
    ${SNDFILE_LIB}
    ${AVFORMAT_LIB}
    ${AVCODEC_LIB}
    ${AVUTIL_LIB}
    ${SWRESAMPLE_LIB}
    ${ONNXRUNTIME_LIB})

# 构建文件夹推理程序
add_executable(kws_folder_inference
    examples/kws_folder_inference.cpp
    ${DETECTOR_LIB_SOURCES})

target_link_libraries(kws_folder_inference
    ${FFTW3_LIB}
    ${FFTW3F_LIB}
    ${SNDFILE_LIB}
    ${AVFORMAT_LIB}
    ${AVCODEC_LIB}
    ${AVUTIL_LIB}
    ${SWRESAMPLE_LIB}
    ${ONNXRUNTIME_LIB})
