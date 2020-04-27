import numpy as np
import cv2
import os
import glob


class MonoVideoOdometry(object):
    def __init__(self, file_path,
                 focal_length=718.8560,
                 pp=(607.1928, 185.2157),
                 lk_params=dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
                 detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
                 camera_extrinsic_rotation=np.identity(3)):
        """
        Arguments:
            file_path {str} -- File path that leads to image sequences or video file
        
        Keyword Arguments:
            focal_length {float} -- Focal length of camera used in image sequence (default: {718.8560})
            pp {tuple} -- Principal point of camera in image sequence (default: {(607.1928, 185.2157)})
            lk_params {dict} -- Parameters for Lucas Kanade optical flow (default: {dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))})
            detector {cv2.FeatureDetector} -- Most types of OpenCV feature detectors (default: {cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)})
        
        Raises:
            ValueError -- Raised when file either file paths are not correct, or img_file_path is not configured correctly
        """

        self.file_path = file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.extrinsic_rotation = camera_extrinsic_rotation
        self.pp = pp
        self.R = np.identity(3)
        self.t = np.zeros(3)
        self.id = 0
        self.n_features = 0
        self.old_frame = None
        self.current_frame = None
        self.good_old = None
        self.good_new = None
        self.p0 = None
        self.p1 = None

        self.frame_paths = glob.glob(os.path.join(file_path, "*.png"))
        self.frame_paths.sort()
        self.n_frames = len(self.frame_paths)

        self.vid_cap = None
        if self.n_frames == 0:  # may be a video file
            self.vid_cap = cv2.VideoCapture(file_path)
            self.n_frames = self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        self.process_frame()

    def has_next_frame(self):
        """
        Used to determine whether there are remaining frames in the folder to process
        
        Returns:
            bool -- Boolean value denoting whether there are still 
            frames in the folder to process
        """
        return self.id < self.n_frames

    def detect(self, img):
        """
        Used to detect features and parse into useable format

        
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        """
        p0 = self.detector.detect(img)
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def visual_odometry(self):
        """
        Used to perform visual odometry. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        """
        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)

        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]

        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        if self.id < 2:
            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
            _, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R, self.t, self.focal, self.pp, None)
        else:
            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
            _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), self.focal, self.pp, None)

            self.t = self.t + self.R.dot(t)
            self.R = R.dot(self.R)

        # Save the total number of good features
        self.n_features = self.good_new.shape[0]

    def get_coordinates(self):
        return np.dot(self.extrinsic_rotation, self.t)

    def process_frame(self):
        """
        Processes images in sequence frame by frame
        """
        if self.id < 2:
            if self.vid_cap is not None:
                success = False
                while not success:
                    success, old_frame = self.vid_cap.read()
                self.old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                success = False
                while not success:
                    success, current_frame = self.vid_cap.read()
                self.current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                self.old_frame = cv2.imread(self.frame_paths[0], 0)
                self.current_frame = cv2.imread(self.frame_paths[1], 0)
            self.visual_odometry()
            if self.vid_cap is not None:
                self.id = self.vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                self.id = 2
        else:
            self.old_frame = self.current_frame
            if self.vid_cap is not None:
                success = False
                while success is False and self.vid_cap.get(cv2.CAP_PROP_POS_FRAMES) < self.n_frames:
                    success, current_frame = self.vid_cap.read()
                if success:
                    self.current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                self.current_frame = cv2.imread(self.frame_paths[self.id], 0)
            self.visual_odometry()
            if self.vid_cap is not None:
                self.id = self.vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                self.id += 1
