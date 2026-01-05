#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <random>
#include <Eigen/Dense>
#include "class_helpers.hpp"

std::vector<int> KMeans_helper(const std::vector<double>& counts) {
    int q = counts.size();
    if (q == 0) return {};

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, q - 1);

    double m1 = counts[dist(mt)];
    double m2 = counts[dist(mt)];
    if (m1 == m2) m2 += 0.01;

    double pm1 = m1, pm2 = m2;
    int iter = 0;

    while ((std::abs(pm1 - m1) > 1e-3 && std::abs(pm2 - m2) > 1e-3) || iter < 50) {
        ++iter;
        pm1 = m1; pm2 = m2;
        int m1s = 0, m2s = 0;
        int m1c = 0, m2c = 0;
        for (int i = 0; i < q; i++) {
            if (std::abs(m1 - counts[i]) < std::abs(m2 - counts[i])) {
                m1s += counts[i];
                m1c += 1;
            }
            else {
                m2s += counts[i];
                m2c += 1;
            }
        }
        m1 = static_cast<double>(m1s) / m1c;
        m2 = static_cast<double>(m2s) / m2c;
    }

    std::vector<int> active_vars;
    if (m1 < m2) {
        int temp = m1;
        m1 = m2;
        m2 = temp;
    }
    for (int i = 0; i < q; i++) {
        if (std::abs(m1 - counts[i]) < std::abs(m2 - counts[i])) active_vars.push_back(i);
    }

    return active_vars;
}
