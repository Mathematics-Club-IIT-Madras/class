#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

inline pair<MatrixXd, VectorXd> iboss_cpp(const MatrixXd &X, const VectorXd &y, int k, bool intercept = false) {
    setNbThreads(1);

    const Index p = X.cols();
    const Index N = X.rows();

    if ((!intercept && p <= 0) || (intercept && p <= 1) || N <= 0) {
        return {MatrixXd(0, p), VectorXd(0)};
    }

    k = min<int>(k, static_cast<int>(N));
    int r = (intercept) ? k / (2 * (p - 1)) : k / (2 * p);
    r = max(r, 1);

    vector<char> global_sel(N, 0);
    #pragma omp parallel
    {
        vector<size_t> local_sel;
        local_sel.reserve(p * 2 * r / omp_get_num_threads() + 100);

        vector<pair<double, size_t>> maximals;
        vector<pair<double, size_t>> minimals;
        maximals.reserve(r + 1);
        minimals.reserve(r + 1);

        #pragma omp for schedule(dynamic)
        for (Index j = (intercept ? 1 : 0); j < p; ++j) {

            maximals.clear();
            minimals.clear();

            for (Index i = 0; i < r; ++i) {
                double val = X(i, j);
                maximals.emplace_back(val, i);
                minimals.emplace_back(val, i);
            }

            make_heap(maximals.begin(), maximals.end(), greater<pair<double, size_t>>());
            make_heap(minimals.begin(), minimals.end());

            for (Index i = r; i < N; ++i) {
                double val = X(i, j);

                if (val > maximals.front().first) {
                    pop_heap(maximals.begin(), maximals.end(), greater<pair<double, size_t>>());
                    maximals.back() = {val, i};
                    push_heap(maximals.begin(), maximals.end(), greater<pair<double, size_t>>());
                }

                if (val < minimals.front().first) {
                    pop_heap(minimals.begin(), minimals.end());
                    minimals.back() = {val, i};
                    push_heap(minimals.begin(), minimals.end());
                }
            }

            for (const auto& kv : maximals) local_sel.push_back(kv.second);
            for (const auto& kv : minimals) local_sel.push_back(kv.second);
        }

        #pragma omp critical
        {
            for(auto idx : local_sel) {
                global_sel[idx] = 1;
            }
        }
    }

    vector<Index> selected_indices;
    selected_indices.reserve(k);

    for(Index i = 0; i < N; ++i) {
        if(global_sel[i]) {
            selected_indices.push_back(i);
        }
    }

    Index actual_k = selected_indices.size();
    MatrixXd X_iboss(actual_k, p);
    VectorXd y_iboss(actual_k);

    #pragma omp parallel for schedule(static)
    for (Index j = 0; j < p; ++j) {
        for (Index i = 0; i < actual_k; ++i) {
            X_iboss(i, j) = X(selected_indices[i], j);
        }
    }

    #pragma omp parallel for schedule(static)
    for (Index i = 0; i < actual_k; ++i) {
        y_iboss(i) = y(selected_indices[i]);
    }

    return {X_iboss, y_iboss};
}
