#include <pybind11/pybind11.h>

#include "main.h"
#include <complex>
#include <cstdio>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// -------------
// pure C++ code
// -------------

int get_num_threads() {
  py::gil_scoped_acquire acquire;
  int n = 0;
#pragma omp parallel
  { n = omp_get_num_threads(); }
  return n;
}

int return1() { return 1; }
double lerp(double frac, double val1, double val2) {
  return (val1 + ((val2 - val1) * frac));
}

// ---------------- 
// Python interface
// ----------------

namespace py = pybind11;
  
typedef py::array_t<double> t_array;
 
PYBIND11_MODULE(_projectors, m) { 
  m.doc() = R"pbdoc(
        module description
        -----------------------

        .. currentmodule:: _projectors 

        .. autosummary::
           :toctree: _generate 

    )pbdoc"; 

  m.def("test_omp", 
        []() {
          py::gil_scoped_release release;
          return get_num_threads();  
        },  
        "Number of OMP threads"); 
  m.def("forward", &forward<double>, "Forward projection");
  m.def("forward_mp", &forward_mp<double>, "Forward projection mp" );
  m.def("forward_mp2", &forward_mp2<double>, "Forward projection mp" );
  // m.def("forward2", &forward2<double>, "Forward projection opt");  
   
  m.def("backward", &backward<double>, "Backward projection");
  m.def("forward_bl", &forward_bl<double>, "Forward projection, bl");
  m.def("backward_bl", &backward_bl<double>, "Backward projection, bl");

  // m.def("testomp1", &testomp1, "testomp1");
  // m.def("testomp2", &testomp2, "testomp2");
  // m.def("testomp3", &testomp3<double>, "testomp3");
  // m.def("testomp4", &testomp4<double>, "testomp4"); 
  // m.def("weighted_kernel", &weighted_kernel<double>, "wrighted kernel");
  // m.def("distribute_convolved_voxel", &distribute_convolved_voxel<double>,
  //       "wrighted kernel"); 
// #ifdef VERSION_INFO
//   m.attr("__version__") = VERSION_INFO; 
// #else
//   m.attr("__version__") = "dev"; 
// #endif  
}
      