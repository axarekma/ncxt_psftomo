#include <cmath>
#include <complex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

inline int clip(int n, int lower, int upper) noexcept {
  return std::max(lower, std::min(n, upper));
}

inline double piece_wise_integrated(double x, const double a, const double b,
                                    const double y_max) {
  if (x < -b)
    return 0.0;
  if (x > b)
    return 1.0;

  auto f1 = [a, b, y_max](double x) -> double {
    return 0.5 * y_max * (x + b) * (x + b) / (b - a);
  };
  auto f2 = [a, b, y_max](double x) -> double {
    return 0.5 * y_max * (b + a) + y_max * x;
  };
  auto f3 = [a, b, y_max](double x) -> double {
    return 0.5 * y_max * (b - x) * (b - x) / (a - b) + 1.0;
  };
  const double a_eps = a + 1e-3;

  if (x > -a_eps && x < a_eps)
    return f2(x);
  if (x < -a)
    return f1(x);
  if (x > a)
    return f3(x);

  // no return value, something wrong
  printf("No case found, this should not happen\n");
  throw std::bad_exception();
}

typedef struct {
  double xmin, xmax, lmax;
} sq_params;

inline sq_params sqpoints(double angle) {
  double cosphi = cos(angle);
  double sinphi = sin(angle);
  double cms = fabs(0.5 * (cosphi - sinphi));
  double cps = fabs(0.5 * (cosphi + sinphi));

  double a = std::min(cms, cps);
  double b = std::max(cms, cps) + 1e-6;
  double lmax = 1 / (((b - a) + 2 * a));
  sq_params retval = {a, b, lmax};
  return retval;
}

template <typename T>
inline py::array_t<double> weighted_kernel(py::array_t<T> psf, int focus,
                                           double const wx0, double const wx1,
                                           double const wx2) {
  auto data_psf = psf.template unchecked<3>();
//  int const psfz = static_cast<int>(data_psf.shape(0));
  int const psfx = static_cast<int>(data_psf.shape(1));
  int const psfy = static_cast<int>(data_psf.shape(2));

  // make weighted kernel
  py::array_t<double> kernel({psfx + 2, psfy});
  auto data_kernel = kernel.template mutable_unchecked<2>();

  for (int xi = 0; xi < data_kernel.shape(0); xi++) {
    for (int yi = 0; yi < data_kernel.shape(1); yi++) {
      data_kernel(xi, yi) = {0};
    }
  }

  // py::print("First element of array is ", data_kernel(0, 0));

  for (int yi = 0; yi < psfy; yi++) {
    for (int xi = 1; xi < psfx + 1; xi++) {
      const T kernel_val = data_psf(focus, xi - 1, yi);
      data_kernel(xi - 1, yi) += wx0 * kernel_val;
      data_kernel(xi, yi) += wx1 * kernel_val;
      data_kernel(xi + 1, yi) += wx2 * kernel_val;
    }
  }
  return kernel;
}

template <typename T>
inline void forward_bl(py::array_t<T> volume, py::array_t<T> projection,
                       py::array_t<T> angles) {
  auto data_x = volume.template unchecked<3>();
  auto data_p = projection.template mutable_unchecked<3>();
  auto data_angles = angles.template unchecked<1>();

  int const nx = static_cast<int>(data_x.shape(0));
  int const ny = static_cast<int>(data_x.shape(1));
  int const nh = static_cast<int>(data_x.shape(2));

  int const pa = static_cast<int>(data_p.shape(0));
  int const px = static_cast<int>(data_p.shape(1));
  int const ph = static_cast<int>(data_p.shape(2));

  if (nh != ph)
    throw std::runtime_error("Volume and projection height must match");
  if (pa != angles.size())
    throw std::runtime_error("Projections stack size must match angles");

  double const xcent = 0.5 * (nx - 1);
  double const ycent = 0.5 * (ny - 1);
  double const pcent = 0.5 * (px - 1);

  // reset projections
  for (int xi = 0; xi < data_p.shape(0); xi++) {
    for (int yi = 0; yi < data_p.shape(1); yi++) {
      for (int zi = 0; zi < data_p.shape(2); zi++) {
        data_p(xi, yi, zi) = {0};
      }
    }
  }

  // #pragma omp parallel for
  for (int ai = 0; ai < angles.size(); ai++) {
    auto const p = sqpoints(angles.at(ai));
    double const cosphi = cos(angles.at(ai));
    double const sinphi = sin(angles.at(ai));

    for (int xi = 0; xi < nx; xi++) {
      for (int yi = 0; yi < ny; yi++) {

        double const xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent;
        int const iqx = static_cast<int>(round(xp));
        double const xv = iqx - xp;

        double const val0 = piece_wise_integrated(xv - 1.5, p.xmin, p.xmax, p.lmax);
        double const val1 = piece_wise_integrated(xv - 0.5, p.xmin, p.xmax, p.lmax);
        double const val2 = piece_wise_integrated(xv + 0.5, p.xmin, p.xmax, p.lmax);
        double const val3 = piece_wise_integrated(xv + 1.5, p.xmin, p.xmax, p.lmax);

        double const wx0 = val1 - val0;
        double const wx1 = val2 - val1;
        double const wx2 = val3 - val2;

        // the pixels where both neighbors excist in projection image
        if (iqx > 0 && iqx < px - 1) {
          for (int zi = 0; zi < nh; zi++) {
            data_p(ai, iqx - 1, zi) += data_x(xi, yi, zi) * wx0;
            data_p(ai, iqx, zi) += data_x(xi, yi, zi) * wx1;
            data_p(ai, iqx + 1, zi) += data_x(xi, yi, zi) * wx2;
          }
        }
        // treat edge cases separately
        // TODO

      } // end yi
    }   // end xi
  }     // end ai

  return;
}

template <typename T>
inline void backward_bl(py::array_t<T> volume, py::array_t<T> projection,
                        py::array_t<T> angles) {
  auto data_x = volume.template mutable_unchecked<3>();
  auto data_p = projection.template unchecked<3>();
  auto data_angles = angles.template unchecked<1>();

  int const nx = static_cast<int>(data_x.shape(0));
  int const ny = static_cast<int>(data_x.shape(1));
  int const nh = static_cast<int>(data_x.shape(2));

  int const pa = static_cast<int>(data_p.shape(0));
  int const px = static_cast<int>(data_p.shape(1));
  int const ph = static_cast<int>(data_p.shape(2));

  if (nh != ph)
    throw std::runtime_error("Volume and projection height must match");
  if (pa != angles.size())
    throw std::runtime_error("Projections stack size must match angles");

  double const xcent = 0.5 * (nx - 1);
  double const ycent = 0.5 * (ny - 1);
  double const pcent = 0.5 * (px - 1);

  // reset volume
  for (int xi = 0; xi < data_x.shape(0); xi++) {
    for (int yi = 0; yi < data_x.shape(1); yi++) {
      for (int zi = 0; zi < data_x.shape(2); zi++) {
        data_x(xi, yi, zi) = {0};
      }
    }
  }

  // #pragma omp parallel for
  for (int ai = 0; ai < angles.size(); ai++) {
    auto const p = sqpoints(angles.at(ai));
    double const cosphi = cos(angles.at(ai));
    double const sinphi = sin(angles.at(ai));

    for (int xi = 0; xi < nx; xi++) {
      for (int yi = 0; yi < ny; yi++) {

        double const xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent;
        int const iqx = static_cast<int>(round(xp));
        double const xv = iqx - xp;

        double const val0 = piece_wise_integrated(xv - 1.5, p.xmin, p.xmax, p.lmax);
        double const val1 = piece_wise_integrated(xv - 0.5, p.xmin, p.xmax, p.lmax);
        double const val2 = piece_wise_integrated(xv + 0.5, p.xmin, p.xmax, p.lmax);
        double const val3 = piece_wise_integrated(xv + 1.5, p.xmin, p.xmax, p.lmax);

        double const wx0 = val1 - val0;
        double const wx1 = val2 - val1;
        double const wx2 = val3 - val2;

        if (iqx > 0 && iqx < px - 1) {
          for (int zi = 0; zi < nh; zi++) {
            // double yp = zi - z_center + py_center;
            // int iqy = static_cast<int>(round(yp));
            int iqy = zi;
            data_x(xi, yi, zi) += wx0 * data_p(ai, iqx - 1, iqy);
            data_x(xi, yi, zi) += wx1 * data_p(ai, iqx, iqy);
            data_x(xi, yi, zi) += wx2 * data_p(ai, iqx + 1, iqy);
          }
        }

      } // end yi
    }   // end xi
  }     // end ai

  return;
}

//

template <typename T>
inline void distribute_convolved_voxel(T val, py::array_t<T> projection,
                                       py::array_t<T> kernel, int xi, int yi, int ai) {
  auto data_p = projection.template mutable_unchecked<3>();
  auto data_kernel = kernel.template unchecked<2>();

  int const px = static_cast<int>(data_p.shape(1));
  int const py = static_cast<int>(data_p.shape(2));
  int const kw_x = static_cast<int>(data_kernel.shape(0));
  int const kw_y = static_cast<int>(data_kernel.shape(1));
  int const kr_x = (kw_x - 1) / 2;
  int const kr_y = (kw_y - 1) / 2;

  for (int ky = std::max(0, kr_y - yi); ky < std::min(py + kr_y - yi, kw_y); ++ky) {
    for (int kx = std::max(0, kr_x - xi); kx < std::min(px + kr_x - xi, kw_x); ++kx) {
      // T old = data_p(xi + (kx - kr_x), yi + (ky - kr_y), ai);
      data_p(ai, xi + (kx - kr_x), yi + (ky - kr_y)) += val * data_kernel(kx, ky);
    }
  }
  return;
}

template <typename T>
inline void forward(py::array_t<T> volume, py::array_t<T> projection,
                    py::array_t<T> psf, py::array_t<T> angles) {
  auto data_x = volume.template unchecked<3>();
  auto data_p = projection.template mutable_unchecked<3>();
  auto data_psf = psf.template unchecked<3>();
  auto data_angles = angles.template unchecked<1>();

  int const nx = static_cast<int>(data_x.shape(0));
  int const ny = static_cast<int>(data_x.shape(1));
  int const nh = static_cast<int>(data_x.shape(2));

  int const pa = static_cast<int>(data_p.shape(0));
  int const px = static_cast<int>(data_p.shape(1));
  int const ph = static_cast<int>(data_p.shape(2));

  // py::print("size_volume ", nx, ny, " Slices:", nh);
  // py::print("size_proj ", "a: ", pa, "(W x H)", px, ph);

  if (nh != ph)
    throw std::runtime_error("Volume and projection height must match");
  if (pa != angles.size())
    throw std::runtime_error("Projections stack size must match angles");

  int const psfz = static_cast<int>(data_psf.shape(0));
//  int const psfx = static_cast<int>(data_psf.shape(1));
  int const psfy = static_cast<int>(data_psf.shape(2));

  double const xcent = 0.5 * (nx - 1);
  double const ycent = 0.5 * (ny - 1);
  double const pcent = 0.5 * (px - 1);
  double const psf_focus = 0.5 * (psfz - 1);
  int const psf_ry = (psfy - 1) / 2;
//  int const psf_rx = (psfx - 1) / 2;

  // reset projections
  for (int xi = 0; xi < data_p.shape(0); xi++) {
    for (int yi = 0; yi < data_p.shape(1); yi++) {
      for (int zi = 0; zi < data_p.shape(2); zi++) {
        data_p(xi, yi, zi) = {0};
      }
    }
  }

  // #pragma omp parallel for
  for (int ai = 0; ai < angles.size(); ai++) {
    auto const p = sqpoints(angles.at(ai));
    double const cosphi = cos(angles.at(ai));
    double const sinphi = sin(angles.at(ai));

    for (int xi = 0; xi < nx; xi++) {
      for (int yi = 0; yi < ny; yi++) {

        double const xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent;
        double const kp = -(xi - xcent) * sinphi + (yi - ycent) * cosphi + psf_focus;
        int const ki = static_cast<int>(round(kp));
        int const iqx = static_cast<int>(round(xp));
        double const xv = iqx - xp;

        double const val0 = piece_wise_integrated(xv - 1.5, p.xmin, p.xmax, p.lmax);
        double const val1 = piece_wise_integrated(xv - 0.5, p.xmin, p.xmax, p.lmax);
        double const val2 = piece_wise_integrated(xv + 0.5, p.xmin, p.xmax, p.lmax);
        double const val3 = piece_wise_integrated(xv + 1.5, p.xmin, p.xmax, p.lmax);

        double const wx0 = val1 - val0;
        double const wx1 = val2 - val1;
        double const wx2 = val3 - val2;

        py::array_t<double> combined_kernel = weighted_kernel(psf, ki, wx0, wx1, wx2);

        // convolve "valid"
        for (int zi = 0; zi < nh; zi++) {

          distribute_convolved_voxel(data_x(xi, yi, zi), projection, combined_kernel,
                                     iqx, zi, ai);
        }

        // extend top and bottom with constant boundary condition
        for (int zi = -psf_ry; zi < 0; zi++) {
          distribute_convolved_voxel(data_x(xi, yi, 0), projection, combined_kernel,
                                     iqx, zi, ai);
        }
        for (int zi = nh; zi < nh + psf_ry; zi++) {
          distribute_convolved_voxel(data_x(xi, yi, nh - 1), projection,
                                     combined_kernel, iqx, zi, ai);
        }

      } // end yi
    }   // end xi
  }     // end ai

  return;
}

template <typename T>
inline T psf_convolved_projection_pixel(py::array_t<T> projection, int xi, int yi,
                                        int zi, py::array_t<T> kernel) {
  auto data_p = projection.template mutable_unchecked<3>();
  auto data_kernel = kernel.template unchecked<2>();

  int const p_w = static_cast<int>(data_p.shape(1));
  int const p_h = static_cast<int>(data_p.shape(2));
  int const kw_x = static_cast<int>(data_kernel.shape(0));
  int const kw_y = static_cast<int>(data_kernel.shape(1));
  int const kr_x = (kw_x - 1) / 2;
  int const kr_y = (kw_y - 1) / 2;
  T element = 0.0;

  if (xi - kr_x >= 0 && yi - kr_y >= 0 && xi + kr_x < p_w && yi + kr_y < p_h) {
    for (int ky = 0; ky < kw_y; ++ky) {
      for (int kx = 0; kx < kw_x; ++kx) {
        element += static_cast<T>(data_p(zi, xi - kr_x + kx, yi - kr_y + ky) *
                                  data_kernel(kx, ky));
      }
    }
    return element;

  } else {
    for (int ky = 0; ky < kw_y; ++ky) {
      int const ind_y = clip(yi + (ky - kr_y), 0, p_h - 1);
      for (int kx = 0; kx < kw_x; ++kx) {
        int const ind_x = clip(xi + (kx - kr_x), 0, p_w - 1);
        element += static_cast<T>(data_p(zi, ind_x, ind_y) * data_kernel(kx, ky));
      }
    }
    return element;
  }
}

template <typename T>
inline void backward(py::array_t<T> volume, py::array_t<T> projection,
                     py::array_t<T> psf, py::array_t<T> angles) {
  auto data_x = volume.template mutable_unchecked<3>();
  auto data_p = projection.template unchecked<3>();
  auto data_psf = psf.template unchecked<3>();
  auto data_angles = angles.template unchecked<1>();

  int const nx = static_cast<int>(data_x.shape(0));
  int const ny = static_cast<int>(data_x.shape(1));
  int const nh = static_cast<int>(data_x.shape(2));

  int const pa = static_cast<int>(data_p.shape(0));
  int const px = static_cast<int>(data_p.shape(1));
  int const ph = static_cast<int>(data_p.shape(2));

  if (nh != ph)
    throw std::runtime_error("Volume and projection height must match");
  if (pa != angles.size())
    throw std::runtime_error("Projections stack size must match angles");

  int const psfz = static_cast<int>(data_psf.shape(0));
//  int const psfx = static_cast<int>(data_psf.shape(1));
//  int const psfy = static_cast<int>(data_psf.shape(2));

  double const xcent = 0.5 * (nx - 1);
  double const ycent = 0.5 * (ny - 1);
  double const pcent = 0.5 * (px - 1);
  double const psf_focus = 0.5 * (psfz - 1);
//  int const psf_ry = (psfy - 1) / 2;
//  int const psf_rx = (psfx - 1) / 2;

  // reset volume
  for (int xi = 0; xi < data_x.shape(0); xi++) {
    for (int yi = 0; yi < data_x.shape(1); yi++) {
      for (int zi = 0; zi < data_x.shape(2); zi++) {
        data_x(xi, yi, zi) = {0};
      }
    }
  }

  // #pragma omp parallel for
  for (int ai = 0; ai < angles.size(); ai++) {
    auto const p = sqpoints(angles.at(ai));
    double const cosphi = cos(angles.at(ai));
    double const sinphi = sin(angles.at(ai));

    for (int xi = 0; xi < nx; xi++) {
      for (int yi = 0; yi < ny; yi++) {

        double const xp = (xi - xcent) * cosphi + (yi - ycent) * sinphi + pcent;
        double const kp = -(xi - xcent) * sinphi + (yi - ycent) * cosphi + psf_focus;
        int const ki = static_cast<int>(round(kp));
        int const iqx = static_cast<int>(round(xp));
        double const xv = iqx - xp;

        double const val0 = piece_wise_integrated(xv - 1.5, p.xmin, p.xmax, p.lmax);
        double const val1 = piece_wise_integrated(xv - 0.5, p.xmin, p.xmax, p.lmax);
        double const val2 = piece_wise_integrated(xv + 0.5, p.xmin, p.xmax, p.lmax);
        double const val3 = piece_wise_integrated(xv + 1.5, p.xmin, p.xmax, p.lmax);

        double const wx0 = val1 - val0;
        double const wx1 = val2 - val1;
        double const wx2 = val3 - val2;

        py::array_t<double> combined_kernel = weighted_kernel(psf, ki, wx0, wx1, wx2);

        for (int zi = 0; zi < nh; zi++) {
          T val =
              psf_convolved_projection_pixel(projection, iqx, zi, ai, combined_kernel);
          data_x(xi, yi, zi) += val;
        }

      } // end yi
    }   // end xi
  }     // end ai

  return;
}
