import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monovideoodometry import MonoVideoOdometry
import os


img_path = 'C:\\Users\\Ali\\Desktop\\Projects\\SLAM\\videos\\data_odometry_gray\\dataset\\sequences\\00\\image_0\\'
pose_path = 'C:\\Users\\Ali\\Desktop\\Projects\\SLAM\\videos\\data_odometry_poses\\dataset\\poses\\00.txt'

focal = 718.8560
pp = (607.1928, 185.2157)
R_total = np.zeros((3, 3))
t_total = np.empty(shape=(3, 1))

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(21, 21),
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))


# Create some random colors
color = np.random.randint(0, 255, (5000, 3))

vo = MonoVideoOdometry(img_path, focal, pp, lk_params)
traj = np.zeros(shape=(600, 800, 3))

while vo.hasNextFrame():
    frame = vo.current_frame

    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == ord("q"):
        break

    vo.process_frame()

    mono_coord = vo.get_mono_coordinates()

    print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))

    draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]

    traj = cv.circle(traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

    cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv.putText(traj, 'Green', (270, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

    cv.imshow('trajectory', traj)
cv.imwrite("./images/trajectory.png", traj)

cv.destroyAllWindows()
