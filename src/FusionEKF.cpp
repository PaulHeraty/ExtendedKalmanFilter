#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  H_laser_ = MatrixXd(2, 4);
  R_laser_ = MatrixXd(2, 2);
  H_radar_ = MatrixXd(3, 4);
  R_radar_ = MatrixXd(3, 3);
  Hj_ = MatrixXd(3, 4);

  /**
  TODO:
    * Finish initializing the FusionEKF.
  */
  // Init H, measurement function
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  H_radar_ << 0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0;

  // Init R, measurement covariance
  R_laser_ << 0.0225, 0,
              0, 0.0225;
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    float x = 0.0;
    float y = 0.0; 
    float xdot = 0.0; 
    float ydot = 0.0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = measurement_pack.raw_measurements_[0]; 
      float phi = measurement_pack.raw_measurements_[1];
      float rodot = measurement_pack.raw_measurements_[2];
      x = ro * cos(phi);
      y = ro * sin(phi);
      xdot = rodot * cos(phi);
      ydot = rodot * sin(phi);
      if (x != 0.0 && y != 0.0) {
        ekf_.x_ << x, y, xdot, ydot;
      }
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x = measurement_pack.raw_measurements_[0];
      y = measurement_pack.raw_measurements_[1];
      xdot = 0.0001;
      ydot = 0.0001;
      if (x != 0.0 && y != 0.0) {
        ekf_.x_ << x, y, xdot, ydot;
      } else {
        ekf_.x_ << 0.0001, 0.0001, 1.0, 1.0;
      }
    }

    // Init P state covariance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
	       0, 1, 0, 0,
	       0, 0, 1000, 0,
	       0, 0, 0, 1000;

    // Init F process model function
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
	       0, 1, 0, 1,
	       0, 0, 1, 0,
	       0, 0, 0, 1;

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;     //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  cout << "dt = " << dt << endl;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
   */
  //1. Modify the F process model matrix so that the time is integrated
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  //2. Set the process noise covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  float noise_ax = 9;
  float noise_ay = 9;
  ekf_.Q_ << ((dt*dt*dt*dt)/4)*noise_ax, 0, ((dt*dt*dt)/2)*noise_ax, 0,
            0, ((dt*dt*dt*dt)/4)*noise_ay, 0, ((dt*dt*dt)/2)*noise_ay,
            ((dt*dt*dt)/2)*noise_ax, 0, dt*dt*noise_ax, 0,
            0, ((dt*dt*dt)/2)*noise_ay, 0, dt*dt*noise_ay;
  //3. Call the Kalman Filter predict() function
  ekf_.Predict();

  cout << "xp_ = " << ekf_.x_ << endl;
  cout << "Pp_ = " << ekf_.P_ << endl;
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
