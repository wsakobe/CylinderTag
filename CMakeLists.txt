cmake_minimum_required(VERSION 3.16)
project(CylinderTag VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Add headers directory
include_directories(${PROJECT_SOURCE_DIR}/header)

# Add source files
add_executable(${PROJECT_NAME} 
    main.cpp
    robust_detector.cpp
    CylinderTag.cpp
    corner_detector.cpp
    pose_estimation.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES})

option(CYLINDERTAG_BUILD_TESTS "Build detector regression tests" OFF)
if(CYLINDERTAG_BUILD_TESTS)
    enable_testing()
    add_executable(cylindertag_regression_tests
        tests/detector_regression.cpp
        robust_detector.cpp
        CylinderTag.cpp
        corner_detector.cpp
        pose_estimation.cpp
    )
    target_link_libraries(cylindertag_regression_tests ${OpenCV_LIBS} ${CERES_LIBRARIES})
    foreach(case threshold_borders right_midpoint invalid_codes column_width stale_detections)
        add_test(NAME ${case} COMMAND cylindertag_regression_tests ${case} ${PROJECT_SOURCE_DIR}/CTag_2f12c.marker)
    endforeach()
endif()

option(CYLINDERTAG_BUILD_ROBUST_TOOLS "Build experimental detector tools" OFF)
if(CYLINDERTAG_BUILD_ROBUST_TOOLS)
    add_executable(cylindertag_video video_detect.cpp robust_detector.cpp
        CylinderTag.cpp corner_detector.cpp pose_estimation.cpp)
    target_link_libraries(cylindertag_video ${OpenCV_LIBS} ${CERES_LIBRARIES})
    if(CYLINDERTAG_BUILD_TESTS)
        add_executable(cylindertag_decoder_test tests/decoder_test.cpp robust_detector.cpp
            CylinderTag.cpp corner_detector.cpp pose_estimation.cpp)
        target_link_libraries(cylindertag_decoder_test ${OpenCV_LIBS} ${CERES_LIBRARIES})
        add_test(NAME robust_dictionary_decoder COMMAND cylindertag_decoder_test ${PROJECT_SOURCE_DIR}/CTag_2f12c.marker)
    endif()
endif()
