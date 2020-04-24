import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

coords = []
while vo.hasNextFrame():
    frame = vo.current_frame

    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == ord("q"):
        break

    vo.process_frame()

    mono_coord = vo.get_mono_coordinates()

    coords.append(mono_coord)

coords = np.array(coords)
fig = plt.figure()  # type: plt.Figure
ax = fig.add_subplot(111, projection='3d')  # type: plt.Axes
ax.plot(*coords.T)
plt.show()

cv.destroyAllWindows()
