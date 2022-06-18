from multiprocessing.connection import wait
import cv2
from cv2 import imshow
import os
from cv2 import waitKey
import numpy as np
import matplotlib.pyplot as plot

window_name = 'frame'
delay = 1
lk_params = dict(
    winSize=(21, 21),           # 検索ウィンドウのサイズ
    maxLevel=3,                 # 追加するピラミッド層数

    # 検索を終了する条件
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    ),

    # 推測値や固有値の使用
    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
)

def extractFeatures(img):
  orb = cv2.ORB_create()
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  # return pts and des
  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def process_frame(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps = cv2.goodFeaturesToTrack(img_gray, 3000, 0.01, 7)
    return kps 

def match_features(prev_corners, corners):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(prev_corners, corners)
    return matches

def estimate_pos():
    print("estimating pos")

def filter_points(prev, new):
    indexs = []

    for p in prev: 
        print(p)


if __name__=="__main__":
    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    F = 525

    camera_matrix = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

    prev_corners = None
    prev_img = None
    result_img = None 
    gray_img = None

    R_f = np.eye(3)
    t_f = np.zeros((3,1))

    xs = []
    ys = []

    while True:
        if cap.isOpened(): 
            ret, frame = cap.read()

            if ret:
                if prev_img is not None:
                    prev_corners = process_frame(prev_img)

                    gray_prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

                    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    next_corners, status, error = cv2.calcOpticalFlowPyrLK(gray_prev_img,gray_img, prev_corners, None, **lk_params)

                    good_old = prev_corners[status == 1]
                    good_new = next_corners[status == 1]
                    result_img = prev_img.copy()

                    essential_mat, _ = cv2.findEssentialMat(good_old, good_new, camera_matrix,cv2.RANSAC)

                    points, R, t, mask = cv2.recoverPose(essential_mat, good_old, good_new)

                    # print(essential_mat)

                    # print("R", R.shape)

                    # print("t", t.shape)

                    t_f = t_f + R_f.dot(t) 
                    R_f = R.dot(R_f)

                    # print(t_f)
                    # print(R_f)

                    xs.append(t_f[0])
                    ys.append(t_f[2])

                    for p, next_p in zip(good_old, good_new):
                        prev_x, prev_y = p.ravel()
                        cur_x, cur_y = next_p.ravel()

                        cv2.circle(result_img, (int(prev_x), int(prev_y)), 3, (0, 255 ,0))
                        cv2.line(result_img, (int(prev_x), int(prev_y)), (int(cur_x), int(cur_y)), (255, 0, 0), 1)

                prev_img = frame 

                if result_img is not None:
                    cv2.imshow(window_name, result_img)

                    if waitKey(0) == ord("q"):
                        break

    plot.plot(xs, ys)
    plot.show()


                    

