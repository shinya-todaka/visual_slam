import cv2
from cv2 import imshow
import os
import numpy as np
import matplotlib.pyplot as plot

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

def get_r_t(F, K, first_image_path, second_image_path):

    img1 = cv2.imread(first_image_path, cv2.IMREAD_COLOR)

    img2 = cv2.imread(second_image_path, cv2.IMREAD_COLOR)

    lk_params = dict( winSize  = (100,100),
         maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    next_features, st, error = cv2.calcOpticalFlowPyrLK(img1, img2, first_features, None , **lk_params)

    good_old = np.int0(first_features[st==1])
    good_new = np.int0(next_features[st==1])

    em, mask = cv2.findEssentialMat(good_new, good_old, 
                                    threshold=0.05, 
                                    prob=0.95, 
                                    focal=F, 
                                    pp=(K[0, 2], K[1, 2]))

    _, R, t, mask2 = cv2.recoverPose(em, good_new, good_old,focal=F, pp=(K[0, 2], K[1, 2]))

    return R, t


def print_camera_parameter(video_path):
            # camera parameters
    cap = cv2.VideoCapture(video_path)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("W: {0}, H: {1}, CNT: {2}".format(W, H, CNT))

if __name__=="__main__":
    
    img = cv2.imread("frames/slam_000.jpg", cv2.IMREAD_COLOR)
    print(img.channels())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray, maxCorners=3000, qualityLevel=0.01, minDistance=7)

