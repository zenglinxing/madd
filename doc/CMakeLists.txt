cmake_minimum_required(VERSION 3.10)

project(Madd-document)

find_program(MKDOCS_EXECUTABLE mkdocs REQUIRED)
message(STATUS "mkdocs: ${MKDOCS_EXECUTABLE}")

file(GLOB_RECURSE MARKDOWN_SOURCE 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/
     "*.md")

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/search.html
    COMMAND ${MKDOCS_EXECUTABLE} build -d ${CMAKE_CURRENT_BINARY_DIR}/doc --dirty
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${MARKDOWN_SOURCE} mkdocs.yml
    COMMENT "Building documentation with MkDocs..."
    VERBATIM
)

add_custom_target(md_doc
    ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/search.html
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc DESTINATION share/madd/)