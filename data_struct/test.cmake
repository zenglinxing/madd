add_executable(test_data-struct-FibonacciHeap-1 data_struct/test/FibonacciHeap-1.c)
target_include_directories(test_data-struct-FibonacciHeap-1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-FibonacciHeap-1 PUBLIC madd)
add_test(NAME DataStruct-FibonacciHeap-1
         COMMAND test_data-struct-FibonacciHeap-1)

add_executable(test_data-struct-FibonacciHeap-2 data_struct/test/FibonacciHeap-2.c)
target_include_directories(test_data-struct-FibonacciHeap-2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-FibonacciHeap-2 PUBLIC madd)
add_test(NAME DataStruct-FibonacciHeap-2
         COMMAND test_data-struct-FibonacciHeap-2)

add_executable(test_data-struct-binary-search-tree-1 data_struct/test/binary-1.c)
target_include_directories(test_data-struct-binary-search-tree-1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-binary-search-tree-1 PUBLIC madd)
add_test(NAME DataStruct-BinarySearchTree-1
         COMMAND test_data-struct-binary-search-tree-1)

add_executable(test_data-struct-binary-search-tree-2 data_struct/test/binary-2.c)
target_include_directories(test_data-struct-binary-search-tree-2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-binary-search-tree-2 PUBLIC madd)
add_test(NAME DataStruct-BinarySearchTree-2
         COMMAND test_data-struct-binary-search-tree-2)

add_executable(test_data-struct-binary-search-tree-3 data_struct/test/binary-3.c)
target_include_directories(test_data-struct-binary-search-tree-3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-binary-search-tree-3 PUBLIC madd)
add_test(NAME DataStruct-BinarySearchTree-3
         COMMAND test_data-struct-binary-search-tree-3)

add_executable(test_data-struct-RBTree-1 data_struct/test/rbtree-1.c)
target_include_directories(test_data-struct-RBTree-1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-RBTree-1 PUBLIC madd)
add_test(NAME DataStruct-RBTree-1
         COMMAND test_data-struct-RBTree-1)

add_executable(test_data-struct-RBTree-2 data_struct/test/rbtree-2.c)
target_include_directories(test_data-struct-RBTree-2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-RBTree-2 PUBLIC madd)
add_test(NAME DataStruct-RBTree-2
         COMMAND test_data-struct-RBTree-2)

add_executable(test_data-struct-RBTree-3 data_struct/test/rbtree-3.c)
target_include_directories(test_data-struct-RBTree-3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-RBTree-3 PUBLIC madd)
add_test(NAME DataStruct-RBTree-3
         COMMAND test_data-struct-RBTree-3)

add_executable(test_data-struct-stack data_struct/test/stack.c)
target_include_directories(test_data-struct-stack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_data-struct-stack PUBLIC madd)
add_test(NAME DataStruct-Stack
         COMMAND test_data-struct-stack)