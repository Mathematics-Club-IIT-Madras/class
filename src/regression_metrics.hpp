#include <Eigen/Dense>

double mean_squared_error(const Eigen::VectorXd &y_true, const Eigen::VectorXd &y_pred);
double r_squared(const Eigen::VectorXd &y_true, const Eigen::VectorXd &y_pred);