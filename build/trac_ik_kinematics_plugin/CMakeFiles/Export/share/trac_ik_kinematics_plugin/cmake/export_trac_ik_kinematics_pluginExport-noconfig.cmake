#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "trac_ik_kinematics_plugin::trac_ik_kinematics_plugin" for configuration ""
set_property(TARGET trac_ik_kinematics_plugin::trac_ik_kinematics_plugin APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(trac_ik_kinematics_plugin::trac_ik_kinematics_plugin PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtrac_ik_kinematics_plugin.so"
  IMPORTED_SONAME_NOCONFIG "libtrac_ik_kinematics_plugin.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS trac_ik_kinematics_plugin::trac_ik_kinematics_plugin )
list(APPEND _IMPORT_CHECK_FILES_FOR_trac_ik_kinematics_plugin::trac_ik_kinematics_plugin "${_IMPORT_PREFIX}/lib/libtrac_ik_kinematics_plugin.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
