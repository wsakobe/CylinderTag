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
        CylinderTag.cpp
        corner_detector.cpp
        pose_estimation.cpp
    )
    target_link_libraries(cylindertag_regression_tests ${OpenCV_LIBS} ${CERES_LIBRARIES})
    foreach(case threshold_borders right_midpoint invalid_codes column_width stale_detections)
        add_test(NAME ${case} COMMAND cylindertag_regression_tests ${case} ${PROJECT_SOURCE_DIR}/CTag_2f12c.marker)
    endforeach()
endif()
