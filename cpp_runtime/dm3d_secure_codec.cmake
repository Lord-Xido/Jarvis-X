# DM3D encrypted byte-stream codec: optional, independent of canonical VM.
option(JARVISX_BUILD_DM3D_SECURE "Build reversible 3D AES-256-GCM codec when dependencies exist" ON)
if(JARVISX_BUILD_DM3D_SECURE)
    if(UNIX AND NOT APPLE)
        find_package(OpenSSL QUIET COMPONENTS Crypto)
        find_package(ZLIB QUIET)
        if(OpenSSL_FOUND AND ZLIB_FOUND)
            add_executable(jarvisx-dm3d-secure
                ${CMAKE_CURRENT_SOURCE_DIR}/dm3d_secure_codec/src/dm3d_secure.cpp
            )
            set_target_properties(jarvisx-dm3d-secure PROPERTIES
                OUTPUT_NAME "DrMoagi-DM3D-Secure"
            )
            target_link_libraries(jarvisx-dm3d-secure PRIVATE OpenSSL::Crypto ZLIB::ZLIB)
            jarvisx_harden(jarvisx-dm3d-secure)
            add_test(NAME dm3d-secure-invertibility
                COMMAND jarvisx-dm3d-secure selftest)
            set_tests_properties(dm3d-secure-invertibility PROPERTIES TIMEOUT 60)
            find_package(Python3 QUIET COMPONENTS Interpreter)
            if(Python3_Interpreter_FOUND)
                add_test(NAME dm3d-secure-roundtrip
                    COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/dm3d_secure_codec/tests/test_integration.py
                        $<TARGET_FILE:jarvisx-dm3d-secure>)
                set_tests_properties(dm3d-secure-roundtrip PROPERTIES TIMEOUT 180)
            endif()
        else()
            message(STATUS "OpenSSL/ZLIB unavailable: optional DM3D secure codec skipped")
        endif()
    else()
        message(STATUS "DM3D secure codec reference targets Linux/POSIX; skipped")
    endif()
endif()
