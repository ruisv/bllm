# FindHobot — locate the generic hobot BPU runtime (hbDNN / hbUCP / HBRT) that
# the NATIVE engine (examples/selfengine.cc) links, INDEPENDENT of libxlm. These
# are the same board libraries bcdl/vision use: headers in /usr/include/hobot,
# libs in /usr/hobot/lib.
#
# Provides:  hobot::dnn  (hbDNN + hbUCP + HBRT4)
#            HOBOT_INCLUDE_DIR / HOBOT_LIB_DIR
#
# The native engine needs none of the OE-LLM SDK — only this. So it can build on
# any board image with the hobot runtime, with or without libxlm present.

set(HOBOT_INCLUDE_ROOT "/usr/include" CACHE PATH "board hobot headers root (contains hobot/)")
set(HOBOT_SYSTEM_LIB_DIR "/usr/hobot/lib" CACHE PATH "board hobot shared libraries directory")

set(_hobot_inc_hints "")
set(_hobot_lib_hints "")
foreach(_env PREFIX BUILD_PREFIX CONDA_PREFIX)
  if(DEFINED ENV{${_env}})
    list(APPEND _hobot_inc_hints "$ENV{${_env}}/include")
    list(APPEND _hobot_lib_hints "$ENV{${_env}}/lib")
  endif()
endforeach()

find_path(HOBOT_INCLUDE_DIR
  NAMES hobot/dnn/hb_dnn.h
  HINTS ${_hobot_inc_hints}
  PATHS ${HOBOT_INCLUDE_ROOT})

find_library(HOBOT_DNN_LIB  NAMES dnn    HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})
find_library(HOBOT_UCP_LIB  NAMES hbucp  HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})
find_library(HOBOT_RT_LIB   NAMES hbrt4  HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hobot
  REQUIRED_VARS HOBOT_INCLUDE_DIR HOBOT_DNN_LIB HOBOT_UCP_LIB HOBOT_RT_LIB)

if(Hobot_FOUND AND NOT TARGET hobot::dnn)
  get_filename_component(HOBOT_LIB_DIR "${HOBOT_DNN_LIB}" DIRECTORY)
  add_library(hobot::dnn UNKNOWN IMPORTED)
  set_target_properties(hobot::dnn PROPERTIES
    IMPORTED_LOCATION "${HOBOT_DNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${HOBOT_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${HOBOT_UCP_LIB};${HOBOT_RT_LIB}")
endif()
