# [QLH-SPLIT] split-stage: serve one stage of a split chain over TCP.
# It links against llama and nothing else -- a stage needs only the public API.
add_executable(split-stage main.cpp)

# the wire protocol lives next to llama_context, so both ends can include it
target_include_directories(split-stage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_link_libraries(split-stage PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})

# Same as every other tool target: llama-split-net.h (shared with llama itself) uses C++17
# nested namespace definitions, and MSVC's default standard is older than that.
target_compile_features(split-stage PRIVATE cxx_std_17)

if (WIN32)
    target_link_libraries(split-stage PRIVATE ws2_32)
endif()

set_target_properties(split-stage PROPERTIES OUTPUT_NAME split-stage)
install(TARGETS split-stage RUNTIME)
