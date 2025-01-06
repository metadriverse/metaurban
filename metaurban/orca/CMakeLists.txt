cmake_minimum_required(VERSION 3.15)

project(multi-agent-navigation)

option(FULL_OUTPUT_FLAG "Enables/disables full output to stdout about reading xml files" OFF)
option(FULL_LOG_FLAG "Enables/disables logging to xml file agents state information" OFF)
option(MAPF_LOG_FLAG "Enables/disables logging to xml files information about MAPF instances in coordinated mode" OFF)


if(FULL_OUTPUT_FLAG)
    add_compile_definitions(FULL_OUTPUT=true)
else(FULL_OUTPUT_FLAG)
    add_compile_definitions(FULL_OUTPUT=false)
endif(FULL_OUTPUT_FLAG)

if(FULL_LOG_FLAG)
    add_compile_definitions(FULL_LOG=true)
else(FULL_LOG_FLAG)
    add_compile_definitions(FULL_LOG=false)
endif(FULL_LOG_FLAG)

if(MAPF_LOG_FLAG)
    add_compile_definitions(MAPF_LOG=true)
else(MAPF_LOG_FLAG)
    add_compile_definitions(MAPF_LOG=false)
endif(MAPF_LOG_FLAG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++14 -w -O0" )


add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)

add_subdirectory(external/tinyxml2)

SET(MA_NAV_LIB_SOURCES
        ${PROJECT_SOURCE_DIR}/src/agent.cpp
        ${PROJECT_SOURCE_DIR}/src/xml_logger.cpp
        ${PROJECT_SOURCE_DIR}/src/mission.cpp
        ${PROJECT_SOURCE_DIR}/src/map.cpp
        ${PROJECT_SOURCE_DIR}/src/xml_reader.cpp
        ${PROJECT_SOURCE_DIR}/src/thetastar.cpp
        ${PROJECT_SOURCE_DIR}/src/geom.cpp
        ${PROJECT_SOURCE_DIR}/src/environment_options.cpp
        ${PROJECT_SOURCE_DIR}/src/direct_planner.cpp
        ${PROJECT_SOURCE_DIR}/src/orca_agent.cpp
        ${PROJECT_SOURCE_DIR}/src/orca_diff_drive_agent.cpp
        ${PROJECT_SOURCE_DIR}/src/agent_pnr.cpp
        ${PROJECT_SOURCE_DIR}/src/sub_map.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf_instances_logger.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/push_and_rotate.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/mapf_actor_set.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/mapf_actor.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/isearch.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/astar.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/search_queue.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/constraints.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/conflict_avoidance_table.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/cbs.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/conflict_set.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/mdd.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/focal_search.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/scipp.cpp
        ${PROJECT_SOURCE_DIR}/src/mapf/sipp.cpp
        ${PROJECT_SOURCE_DIR}/src/agent_pnr_ecbs.cpp
        ${PROJECT_SOURCE_DIR}/src/agent_returning.cpp)

add_library(ma_navigation_lib STATIC ${MA_NAV_LIB_SOURCES})
target_include_directories(ma_navigation_lib PUBLIC ${PROJECT_SOURCE_DIR}/include/)


target_link_libraries(ma_navigation_lib tinyxml2)

pybind11_add_module(bind src/experiments/bind.cpp ${MA_NAV_LIB_SOURCES})
target_include_directories(bind PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_link_libraries(bind PRIVATE tinyxml2)
