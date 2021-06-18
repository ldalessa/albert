add_executable(test test.cpp)
target_link_libraries(test PRIVATE albert::albert)

add_executable(basic_tensor basic_tensor.cpp)
target_link_libraries(basic_tensor PRIVATE albert::albert)

add_executable(basic_expressions basic_expressions.cpp)
target_link_libraries(basic_expressions PRIVATE albert::albert)
