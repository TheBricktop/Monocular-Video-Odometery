import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from monovideoodometry import MonoVideoOdometry
import os


img_path = 'data/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/'

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

vo = MonoVideoOdometry(img_path, focal, pp, lk_params, camera_extrinsic_rotation=camera_extrinsics)

coords = []
cv.namedWindow("frame", cv.WINDOW_NORMAL)
while vo.has_next_frame():
    frame = vo.current_frame

    disp_frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    for i in range(len(vo.good_old)):
        cv.line(disp_frame, tuple(vo.good_old[i]), tuple(vo.good_new[i]), (255, 255, 0))

    cv.imshow('frame', disp_frame)
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
