# Patch liboci.h to fix clang compatibility on Windows
# This script is called by CMakeLists.txt after Go generates the header

if(NOT DEFINED HEADER_FILE)
    message(FATAL_ERROR "HEADER_FILE not defined")
endif()

if(NOT EXISTS "${HEADER_FILE}")
    message(FATAL_ERROR "Header file does not exist: ${HEADER_FILE}")
endif()

# Read the header file
file(READ "${HEADER_FILE}" HEADER_CONTENT)

# Replace the problematic section
# The issue is that clang on Windows defines _MSC_VER but doesn't support _Fcomplex/_Dcomplex
# We need to check for __clang__ and use C++ complex types in that case

string(REPLACE 
    "#ifdef _MSC_VER\n#if !defined(__cplusplus) || _MSVC_LANG <= 201402L\n#include <complex.h>\ntypedef _Fcomplex GoComplex64;\ntypedef _Dcomplex GoComplex128;\n#else\n#include <complex>\ntypedef std::complex<float> GoComplex64;\ntypedef std::complex<double> GoComplex128;\n#endif"
    "#ifdef _MSC_VER\n#if defined(__clang__) || (defined(__cplusplus) && _MSVC_LANG > 201402L)\n#include <complex>\ntypedef std::complex<float> GoComplex64;\ntypedef std::complex<double> GoComplex128;\n#elif !defined(__cplusplus) || _MSVC_LANG <= 201402L\n#include <complex.h>\ntypedef _Fcomplex GoComplex64;\ntypedef _Dcomplex GoComplex128;\n#else\n#include <complex>\ntypedef std::complex<float> GoComplex64;\ntypedef std::complex<double> GoComplex128;\n#endif"
    HEADER_CONTENT
    "${HEADER_CONTENT}"
)

# Write the patched header back
file(WRITE "${HEADER_FILE}" "${HEADER_CONTENT}")

message(STATUS "Successfully patched ${HEADER_FILE}")
