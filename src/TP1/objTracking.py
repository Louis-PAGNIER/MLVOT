from KalmanFilter import KalmanFilter
from Detector import detect
import cv2

kf = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

cap = cv2.VideoCapture('randomball.avi')

ret = True
history = []
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    center = detect(frame)[0]
    cv2.circle(frame, (int(center[0]), int(center[1])), 10, (0, 255, 0), 2)

    kf.predict()
    kf.update(center)

    predict = (int(kf.x_k_[0]), int(kf.x_k_[1]))
    cv2.rectangle(frame, (predict[0] - 15, predict[1] - 15), (predict[0] + 15, predict[1] + 15), (255, 0, 0), 2)
    estimate = (int(kf.x_k[0]), int(kf.x_k[1]))
    cv2.rectangle(frame, (estimate[0] - 15, estimate[1] - 15), (estimate[0] + 15, estimate[1] + 15), (0, 0, 255), 2)

    if len(history) > 0:
        for i in range(len(history) - 1):
            cv2.line(frame, history[i], history[i + 1], (255, 0, 0), 2)

    history.append(estimate)

    cv2.imshow('frame', frame)
    cv2.waitKey(50)

