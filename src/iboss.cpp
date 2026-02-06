#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>

inline void IBOSS(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, Eigen::MatrixXd &X_iboss, Eigen::VectorXd &y_iboss, int k, bool intercept) {
    
    const Eigen::Index p = X.cols();
    const Eigen::Index N = X.rows();

    if ((!intercept && p <= 0) || (intercept && p <= 1) || N <= 0) {
        return;
    }

    k = std::min<int>(k, static_cast<int>(N));
    int r = (intercept) ? k / (2 * (p - 1)) : k / (2 * p);
    r = std::max(r, 1);

    std::vector<char> global_sel(N, 0);
    #pragma omp parallel
    {
        std::vector<size_t> local_sel;
        local_sel.reserve(p * 2 * r / omp_get_num_threads() + 100);

        std::vector<std::pair<double, size_t>> maximals;
        std::vector<std::pair<double, size_t>> minimals;
        maximals.reserve(r + 1);
        minimals.reserve(r + 1);

        #pragma omp for schedule(dynamic)
        for (Eigen::Index j = (intercept ? 1 : 0); j < p; ++j) {

            maximals.clear();
            minimals.clear();

            for (Eigen::Index i = 0; i < r; ++i) {
                double val = X(i, j);
                maximals.emplace_back(val, i);
                minimals.emplace_back(val, i);
            }

            std::make_heap(maximals.begin(), maximals.end(), std::greater<std::pair<double, size_t>>());
            std::make_heap(minimals.begin(), minimals.end());

            for (Eigen::Index i = r; i < N; ++i) {
                double val = X(i, j);

                if (val > maximals.front().first) {
                    std::pop_heap(maximals.begin(), maximals.end(), std::greater<std::pair<double, size_t>>());
                    maximals.back() = {val, i};
                    std::push_heap(maximals.begin(), maximals.end(), std::greater<std::pair<double, size_t>>());
                }

                if (val < minimals.front().first) {
                    std::pop_heap(minimals.begin(), minimals.end());
                    minimals.back() = {val, i};
                    std::push_heap(minimals.begin(), minimals.end());
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

    std::vector<Eigen::Index> selected_indices;
    selected_indices.reserve(k);

    for(Eigen::Index i = 0; i < N; ++i) {
        if(global_sel[i]) {
            selected_indices.push_back(i);
        }
    }

    Eigen::Index actual_k = selected_indices.size();
    X_iboss.resize(actual_k, p);
    y_iboss.resize(actual_k);

    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < p; ++j) {
        for (Eigen::Index i = 0; i < actual_k; ++i) {
            X_iboss(i, j) = X(selected_indices[i], j);
        }
    }

    #pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < actual_k; ++i) {
        y_iboss(i) = y(selected_indices[i]);
    }

    return;
}
