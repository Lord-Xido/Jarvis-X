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
file(SHA256 "${SOURCE}" source_hash)
file(SHA256 "${GENERATED}" generated_hash)
file(REMOVE "${GENERATED}")
if(NOT source_hash STREQUAL generated_hash)
    message(FATAL_ERROR "quine output differs from source: ${source_hash} vs ${generated_hash}")
endif()
message(STATUS "Investigative3D quine reproduced exact source (SHA256 ${source_hash})")
