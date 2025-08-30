cmake_minimum_required(VERSION 3.10)

project(Madd-document)

#[[find_package(LATEX COMPONENTS PDFLATEX BIBTEX REQUIRED)

set(TEX_SOURCES madd-guide.tex)

file(GLOB_RECURSE TEX_CHAP_SOURCE 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/chap
     "*.tex")
message(STATUS "tex chap: ${TEX_CHAP_SOURCE}")
set(TEX_CHAP_SRC_LIST)
foreach(chap_tex ${TEX_CHAP_SOURCE})
    #get_filename_component(chap_filename "${chap_tex}" NAME)
    #list(APPEND TEX_CHAP_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/chap/${chap_filename})
    list(APPEND TEX_CHAP_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/chap/${chap_tex})
endforeach()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/madd-guide.pdf
    COMMAND ${PDFLATEX_COMPILER} -output-directory=${CMAKE_CURRENT_BINARY_DIR} -interaction=nonstopmode madd-guide.tex
    COMMAND ${BIBTEX_COMPILER} madd-guide
    COMMAND ${PDFLATEX_COMPILER} -output-directory=${CMAKE_CURRENT_BINARY_DIR} -interaction=nonstopmode madd-guide.tex
    COMMAND ${PDFLATEX_COMPILER} -output-directory=${CMAKE_CURRENT_BINARY_DIR} -interaction=nonstopmode madd-guide.tex
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${TEX_SOURCES} ${TEX_CHAP_SRC_LIST}
    COMMENT "Building PDF document with BibTeX"
)

add_custom_target(madd_latex_pdf ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/madd-guide.pdf)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/madd-guide.pdf
    DESTINATION share/doc/madd
)
]]#

find_program(MKDOCS_EXECUTABLE mkdocs REQUIRED)
message(STATUS "mkdocs: ${MKDOCS_EXECUTABLE}")

file(GLOB_RECURSE MARKDOWN_SOURCE 
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/
     "*.md")

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/search.html
    COMMAND ${MKDOCS_EXECUTABLE} build -d ${CMAKE_CURRENT_BINARY_DIR}/doc --dirty
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building documentation with MkDocs..."
    VERBATIM
)

add_custom_target(md_doc
    ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/search.html
)