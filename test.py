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
camera_extrinsics = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=float)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(21, 21),
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))


# Create some random colors
color = np.random.randint(0, 255, (5000, 3))

vo = MonoVideoOdometry(img_path, focal, pp, lk_params, camera_extrinsics=camera_extrinsics)

coords = []
while vo.hasNextFrame():
    frame = vo.current_frame

    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == ord("q"):
        break

    vo.process_frame()

    coords.append(vo.get_coordinates())

coords = np.array(coords).T
fig = plt.figure()  # type: plt.Figure
ax = fig.add_subplot(111, projection='3d')  # type: plt.Axes
ax.plot(*coords)
xy_min = np.min(coords[[0, 1]])
xy_max = np.max(coords[[0, 1]])
xy_span = xy_max-xy_min
z_min = np.min(coords[2])
z_max = np.max(coords[2])
z_mid = (z_max + z_min) / 2
z_exaggeration = 10
ax.set_xlim(xy_min, xy_max)
ax.set_ylim(xy_min, xy_max)
ax.set_zlim(z_mid-xy_span/z_exaggeration, z_mid+xy_span/z_exaggeration)
plt.show()

cv.destroyAllWindows()
