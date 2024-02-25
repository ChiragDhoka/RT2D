# CMakeList.txt : CMake project for RT3D, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (RT3D "main.cpp" "main.h" "knn.h" "processing.h" "csv_util.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET RT3D PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
