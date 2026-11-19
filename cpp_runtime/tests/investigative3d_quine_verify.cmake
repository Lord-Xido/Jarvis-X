if(NOT DEFINED PROGRAM OR NOT DEFINED SOURCE OR NOT DEFINED GENERATED)
    message(FATAL_ERROR "PROGRAM, SOURCE and GENERATED must be provided")
endif()
execute_process(
    COMMAND "${PROGRAM}" --print-source
    OUTPUT_FILE "${GENERATED}"
    RESULT_VARIABLE quine_status
)
if(NOT quine_status EQUAL 0)
    message(FATAL_ERROR "quine executable failed: ${quine_status}")
endif()
if(CMAKE_HOST_WIN32)
    # Git checkout and stdio can translate newlines independently on Windows.
    file(READ "${SOURCE}" source_text)
    file(READ "${GENERATED}" generated_text)
    string(REPLACE "\r\n" "\n" source_text "${source_text}")
    string(REPLACE "\r\n" "\n" generated_text "${generated_text}")
    string(SHA256 source_hash "${source_text}")
    string(SHA256 generated_hash "${generated_text}")
else()
    # On POSIX platforms test literal byte-for-byte equivalence.
    file(SHA256 "${SOURCE}" source_hash)
    file(SHA256 "${GENERATED}" generated_hash)
endif()
file(REMOVE "${GENERATED}")
if(NOT source_hash STREQUAL generated_hash)
    message(FATAL_ERROR "quine differs from source: ${source_hash} vs ${generated_hash}")
endif()
message(STATUS "Investigative3D quine verified (SHA256 ${source_hash})")
