add_executable(test test.cpp)
target_link_libraries(test PRIVATE albert::albert)

add_executable(tensor tensor.cpp)
target_link_libraries(tensor PRIVATE albert::albert)

add_executable(expressions expressions.cpp)
target_link_libraries(expressions PRIVATE albert::albert)
