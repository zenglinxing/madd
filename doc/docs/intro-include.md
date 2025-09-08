Include & Link
===

This section introduces how to include and link Madd library in your library and program.

In your C/C++ source

```C
#include<madd/madd.h>
```

Link Madd when building

```bash
gcc main.c -o main -lmadd
```

Link through CMake
---

If you want to use Madd in your CMake project.

```cmake
find_package(madd)
# print the relevant variables
message(STATUS "madd library: ${madd_LIBRARIES}")
message(STATUS "madd include dir: ${madd_INCLUDE_DIRS}")
message(STATUS "madd library dir: ${madd_LIBRARY_DIRS}")

# in your target
target_link_libraries(your_target PUBLIC ${madd_LIBRARIES})
target_include_directories(your_target PUBLIC ${madd_INCLUDE_DIRS}")
target_link_direcotries(your_target PUBLIC ${madd_LIBRARY_DIRS})
```