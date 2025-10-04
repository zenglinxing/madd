add_executable(test_sort-merge sort/test/sort-merge.c)
target_include_directories(test_sort-merge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort-merge PUBLIC madd)
add_test(NAME Sort-Merge
         COMMAND test_sort-merge)

add_executable(test_sort-counting sort/test/sort-counting.c)
target_include_directories(test_sort-counting PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort-counting PUBLIC madd)
add_test(NAME Sort-Counting
         COMMAND test_sort-counting)

add_executable(test_sort_binary-search sort/test/binary-search.c)
target_include_directories(test_sort_binary-search PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort_binary-search PUBLIC madd)
add_test(NAME Sort-BinarySearchFunc
         COMMAND test_sort_binary-search)

add_executable(test_sort_binary-search-insert sort/test/binary-search-insert.c)
target_include_directories(test_sort_binary-search-insert PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort_binary-search-insert PUBLIC madd)
add_test(NAME Sort-BinarySearchInsertFunc
         COMMAND test_sort_binary-search-insert)

add_executable(test_sort-heap sort/test/sort-heap.c)
target_include_directories(test_sort-heap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort-heap PUBLIC madd)
add_test(NAME Sort-Heap
         COMMAND test_sort-heap)

add_executable(test_sort-insertion sort/test/sort-insertion.c)
target_include_directories(test_sort-insertion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort-insertion PUBLIC madd)
add_test(NAME Sort-Insertion
         COMMAND test_sort-insertion)

add_executable(test_sort-quicksort sort/test/sort-quicksort.c)
target_include_directories(test_sort-quicksort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort-quicksort PUBLIC madd)
add_test(NAME Sort-Quicksort
         COMMAND test_sort-quicksort)

add_executable(test_sort-shell sort/test/sort-shell.c)
target_include_directories(test_sort-shell PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_sort-shell PUBLIC madd)
add_test(NAME Sort-ShellSort
         COMMAND test_sort-shell)