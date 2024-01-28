import numpy as np

class KalmanFilter:

    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas

        self.u = np.array([[u_x], [u_y]])
        self.x_k = np.array([[0], [0], [0], [0]])
        print(self.x_k)

        self.A = np.array( [[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.B = np.array( [[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])

        self.H = np.array( [[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.array( [[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * self.std_acc**2

        self.R = np.array( [[self.x_std_meas**2, 0],
                            [0, self.y_std_meas**2]])

        self.P = np.identity(n=self.A.shape[0])

    def predict(self):
        self.x_k_ = self.A @ self.x_k + self.B @ self.u
        print(self.x_k_)
        self.P_ = self.A @ self.P @ self.A.T + self.Q

    def update(self, z_k):
        S = self.H @ self.P_ @ self.H.T + self.R
        K = self.P_ @ self.H.T @ np.linalg.inv(S)
        self.x_k = self.x_k_ + K @ (z_k - self.H @ self.x_k_)
        self.P = (np.identity(n=self.P.shape[0]) - K @ self.H) @ self.P_

