#include "subselectR_control.h"
#include "SRHT.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <bit>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace Eigen;

void multiply(double* __restrict y, int n)
{
    if (n <= 2048)
    {
        for (int h = 1; h < n; h = h * 2)
        {
            for (int i = 0; i < n; i += 2 * h)
            {
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (int j = i; j < i + h; j++)
                {
                    double s = y[j];
                    double t = y[j + h];
                    y[j] = s + t;
                    y[j + h] = s - t;
                }
            }
        }
        return;
    }

    int h = n / 2;
    multiply(y, h);
    multiply(y + h, h);

    // Cache-locality block tiling combining sub-halves
    constexpr int B = 1024;
    for (int i0 = 0; i0 < h; i0 += B)
    {
        int limit = std::min(i0 + B, h);
#ifdef _OPENMP
        #pragma omp simd
#endif
        for (int i = i0; i < limit; ++i)
        {
            double s = y[i];
            double t = y[i + h];
            y[i] = s + t;
            y[i + h] = s - t;
        }
    }
}

std::pair<MatrixXd, VectorXd> SRHT(
    const MatrixXd& A,
    const VectorXd& b,
    const int& r,
    std::optional<uint64_t> seed)
{
    // Constant factor sqrt(n/r) cancels in regression calculations.
    int n = static_cast<int>(A.rows());
    int d = static_cast<int>(A.cols());
    unsigned int n_padded = bit_ceil((unsigned)n);

    std::mt19937_64 mt;
    if (seed.has_value()) {
        mt.seed(seed.value());
    } else {
        std::random_device rd;
        mt.seed((static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd()));
    }

    uniform_int_distribution<int> dist(0, static_cast<int>(n_padded - 1));
    vector<int> S(static_cast<std::size_t>(r));
    for (int i = 0; i < r; ++i)
    {
        S[static_cast<std::size_t>(i)] = dist(mt);
    }

    // Fast random sign generation: consume 64-bit random words via bit-shifts
    VectorXd diagonal(n);
    int idx = 0;
    while (idx < n)
    {
        uint64_t bits = mt();
        int count = std::min(64, n - idx);
        for (int b_idx = 0; b_idx < count; ++b_idx)
        {
            diagonal(idx++) = (bits & 1ULL) ? 1.0 : -1.0;
            bits >>= 1;
        }
    }

    VectorXd temp_b = VectorXd::Zero(n_padded);
    temp_b.head(n) = b;
    temp_b.head(n).array() *= diagonal.array();
    multiply(temp_b.data(), n_padded);

    sort(S.begin(), S.end());
    VectorXd y_f(r);
    for (int i = 0; i < r; i++)
    {
        y_f(i) = temp_b(S[static_cast<std::size_t>(i)]);
    }

    MatrixXd X_f(r, d);
#ifdef _OPENMP
    #pragma omp parallel num_threads(subselectR_threads())
#endif
    {
        VectorXd temp = VectorXd::Zero(n_padded);
#ifdef _OPENMP
        #pragma omp for
#endif
        for (int j = 0; j < d; j++)
        {
            temp.tail(n_padded - n).setZero();
            const double* col_ptr = &A(0, j);
            double* temp_ptr = temp.data();
            const double* diag_ptr = diagonal.data();

            for (int i = 0; i < n; i++)
            {
                temp_ptr[i] = col_ptr[i] * diag_ptr[i];
            }

            multiply(temp.data(), n_padded);

            for (int i = 0; i < r; i++)
            {
                X_f(i, j) = temp(S[static_cast<std::size_t>(i)]);
            }
        }
    }
    return {X_f, y_f};
}
