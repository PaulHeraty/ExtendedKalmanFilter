#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;	// state/position
  P_ = P_in;	// state/position noise/covariance
  F_ = F_in;	// state transition function, Process model matrix
  H_ = H_in;	// Measurement function
  R_ = R_in;	// process model noise/variance
  Q_ = Q_in;	// sensor measurement noise/covariance
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;			// prior
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;	// state variance
}

void KalmanFilter::Update(const VectorXd &z) {  // z is sensor measurement
  VectorXd z_pred = H_ * x_;	// convert prior to measurement space
  VectorXd y = z - z_pred;	// residual
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;	// system uncertainty
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;	// Kalman gain
  x_ = x_ + (K * y);		// posterior
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;	// posterior variance
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd hx(3);
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  hx << sqrt(px*px + py*py), 
                 atan(py/px),
                (px*vx + py*vy) / sqrt(px*px + py*py);
  VectorXd z_pred = hx;	// convert prior to measurement space
  VectorXd y = z - z_pred;	// residual
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;	// system uncertainty
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;	// Kalman gain
  x_ = x_ + (K * y);		// posterior
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;	// posterior variance
}
