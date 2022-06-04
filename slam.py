import imp
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

def playVideo(path):
    cap = cv2.VideoCapture(path)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("cap is not opend!!")
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__=="__main__":
    
    img1 = cv2.imread("frames/slam_000.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    first_features = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 7)

    img2 = cv2.imread("frames/slam_010.jpg", cv2.IMREAD_COLOR)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    second_features = cv2.goodFeaturesToTrack(gray2, 3000, 0.01, 7)

    playVideo("test_kitti984.mp4")

    F = 525 
    W = 1920 
    H = 1080

    K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

    R, t = get_r_t(F, K, "frames/slam_000.jpg", "frames/slam_001.jpg")

    xs = []
    ys = []

    for i in range(2, 20):
        R_next, t_next = get_r_t(F, K, "frames/slam_{}.jpg".format(format(i + 1, "03")), "frames/slam_{}.jpg".format(format(i, "03")),)
        
        print(t_next)

        t = t + (R.dot(t_next))

        R = R_next.dot(R)

        x, y = t[0], t[1]

        xs.append(x)
        ys.append(y)

        print("x: {0}, y: {1}".format(x, y))

    R, t = get_r_t(F, K, "frames/slam_000.jpg", "frames/slam_001.jpg")

    image_mask = np.zeros_like(img1)

    plot.plot(xs, ys)
    plot.show()

    # for i, (old,new) in enumerate(zip(good_old, good_new)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     image_mask = cv2.line(image_mask, (a,b),(c,d), (0,0,255), 2)
    #     frame = cv2.circle(img1,(a,b),5, (255,0,0),-1)

    # f = 500
    # E = cv2.findEssentialMat(good_new, good_old,)

    # img = cv2.add(frame, image_mask)

    # cv2.imshow('frame',img)
    # cv2.waitKey(5000)

