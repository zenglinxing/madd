add_executable(test_data-struct-FibonacciHeap-1 data_struct/test/FibonacciHeap-1.c)
target_include_directories(test_data-struct-FibonacciHeap-1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-FibonacciHeap-1 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-FibonacciHeap-1 PUBLIC madd)
add_test(NAME DataStruct-FibonacciHeap-1
         COMMAND test_data-struct-FibonacciHeap-1)

add_executable(test_data-struct-FibonacciHeap-2 data_struct/test/FibonacciHeap-2.c)
target_include_directories(test_data-struct-FibonacciHeap-2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-FibonacciHeap-2 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-FibonacciHeap-2 PUBLIC madd)
add_test(NAME DataStruct-FibonacciHeap-2
         COMMAND test_data-struct-FibonacciHeap-2)

add_executable(test_data-struct-binary-search-tree-1 data_struct/test/binary-1.c)
target_include_directories(test_data-struct-binary-search-tree-1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-binary-search-tree-1 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-binary-search-tree-1 PUBLIC madd)
add_test(NAME DataStruct-BinarySearchTree-1
         COMMAND test_data-struct-binary-search-tree-1)

add_executable(test_data-struct-binary-search-tree-2 data_struct/test/binary-2.c)
target_include_directories(test_data-struct-binary-search-tree-2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-binary-search-tree-2 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-binary-search-tree-2 PUBLIC madd)
add_test(NAME DataStruct-BinarySearchTree-2
         COMMAND test_data-struct-binary-search-tree-2)

add_executable(test_data-struct-binary-search-tree-3 data_struct/test/binary-3.c)
target_include_directories(test_data-struct-binary-search-tree-3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-binary-search-tree-3 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-binary-search-tree-3 PUBLIC madd)
add_test(NAME DataStruct-BinarySearchTree-3
         COMMAND test_data-struct-binary-search-tree-3)

add_executable(test_data-struct-RBTree-1 data_struct/test/rbtree-1.c)
target_include_directories(test_data-struct-RBTree-1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-RBTree-1 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-RBTree-1 PUBLIC madd)
add_test(NAME DataStruct-RBTree-1
         COMMAND test_data-struct-RBTree-1)

add_executable(test_data-struct-RBTree-2 data_struct/test/rbtree-2.c)
target_include_directories(test_data-struct-RBTree-2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-RBTree-2 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-RBTree-2 PUBLIC madd)
add_test(NAME DataStruct-RBTree-2
         COMMAND test_data-struct-RBTree-2)

add_executable(test_data-struct-RBTree-3 data_struct/test/rbtree-3.c)
target_include_directories(test_data-struct-RBTree-3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-RBTree-3 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-RBTree-3 PUBLIC madd)
add_test(NAME DataStruct-RBTree-3
         COMMAND test_data-struct-RBTree-3)

add_executable(test_data-struct-RBTree-Deepseek data_struct/test/rbtree-deepseek.c)
target_include_directories(test_data-struct-RBTree-Deepseek PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-RBTree-Deepseek PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-RBTree-Deepseek PUBLIC madd)
add_test(NAME DataStruct-RBTreeDeepseek
         COMMAND test_data-struct-RBTree-Deepseek)

add_executable(test_data-struct-stack data_struct/test/stack.c)
target_include_directories(test_data-struct-stack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-stack PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-stack PUBLIC madd)
add_test(NAME DataStruct-Stack
         COMMAND test_data-struct-stack)

add_executable(test_data-struct-queue data_struct/test/queue.c)
target_include_directories(test_data-struct-queue PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-queue PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-queue PUBLIC madd)
add_test(NAME DataStruct-Queue
         COMMAND test_data-struct-queue)

add_executable(test_data-struct-queue-minimal data_struct/test/queue-minimal.c)
target_include_directories(test_data-struct-queue-minimal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_data-struct-queue-minimal PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_data-struct-queue-minimal PUBLIC madd)
add_test(NAME DataStruct-QueueMinimal
         COMMAND test_data-struct-queue-minimal)