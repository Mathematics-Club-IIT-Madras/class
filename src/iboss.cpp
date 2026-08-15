#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <numeric>
#include <omp.h>

#include "iboss.h"

void IBOSS(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, 
                    Eigen::MatrixXd &X_iboss, Eigen::VectorXd &y_iboss, int k) {
    
    const Eigen::Index p = X.cols();
    const Eigen::Index N = X.rows();

    if (p <= 0 || N <= 0) {
        return;
    }

    k = std::min<int>(k, static_cast<int>(N));
    int num_features = p;
    
    int r = k / (2 * num_features);
    if (r == 0) r = 1;

    int u = std::min<int>(k, 2 * num_features * r); 

    std::vector<int> nums(N);
    std::iota(nums.begin(), nums.end(), 0);

    int current_offset = 0;

    for (Eigen::Index j = 0; j < p; ++j) {
        if (current_offset >= u) break;

        auto greater = [&X, j](int a, int b) { return X(a, j) > X(b, j); };
        auto lesser = [&X, j](int a, int b) { return X(a, j) < X(b, j); };

        int r_max = std::min<int>(r, u - current_offset);
        if (r_max > 0) {
            std::nth_element(nums.begin() + current_offset, 
                             nums.begin() + current_offset + r_max - 1, 
                             nums.end(), greater);
            current_offset += r_max;
        }
        
        if (current_offset >= u) break;

        int r_min = std::min<int>(r, u - current_offset);
        if (r_min > 0) {
            std::nth_element(nums.begin() + current_offset, 
                             nums.begin() + current_offset + r_min - 1, 
                             nums.end(), lesser);
            current_offset += r_min;
        }
    }

    X_iboss.resize(current_offset, p);
    y_iboss.resize(current_offset);

    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < p; ++j) {
        for (int i = 0; i < current_offset; ++i) {
            X_iboss(i, j) = X(nums[i], j);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < current_offset; ++i) {
        y_iboss(i) = y(nums[i]);
    }
}