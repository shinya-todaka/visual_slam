import cv2
from matplotlib.pyplot import flag
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, matrix_rank
from constants import RANSAC_RESIDUAL_THRES, RANSAC_MAX_TRIALS
from skimage.measure import ransac
from helpers import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform

def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret


def fundamentalToRt(F):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,d,Vt = np.linalg.svd(F)
  if np.linalg.det(U) < 0:
    U *= -1.0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]

  # TODO: Resolve ambiguities in better ways. This is wrong.
  if t[2] < 0:
    t *= -1
  
  return np.linalg.inv(poseRt(R, t))

def extract_features(img):
  orb = cv2.ORB_create()
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def keypoints_to_ndarray(keypoints):
    kps = []
    for keypoint in keypoints: 
        kps.append(keypoint.pt)
    return np.array([kps])

def match_frames(kps1, kps2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()

    for m,n in matches:
        if m.distance < 0.75*n.distance:

            p1 = kps1[m.queryIdx]
            p2 = kps2[m.trainIdx]

            # be within orb distance 32
            if m.distance < 32:
                # keep around indices
                # TODO: refactor this to not be O(N^2)
                if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idx1s.add(m.queryIdx)
                    idx2s.add(m.trainIdx)
                    ret.append((p1, p2))

    # no duplicates
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # fit matrix
    # model, inliers = ransac((ret[:, 0], ret[:, 1]),
    #                         EssentialMatrixTransform,
    #                         min_samples=8,
    #                         residual_threshold=RANSAC_RESIDUAL_THRES,
    #                         max_trials=RANSAC_MAX_TRIALS)

    F, mask = cv2.findFundamentalMat(ret[:,0] ,ret[:, 1],cv2.RANSAC)

    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=100)

    print("Matches:  %d -> %d -> %d -> %d" % (len(des1), len(matches), len(inliers), sum(inliers)))
    return ret[:, 0][inliers], ret[:, 1][inliers], fundamentalToRt(model.params)

def getRT(frame, ref_frame):
    # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ref_gray_image = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    kps1, des1 = extract_features(frame)

    kps2, des2 = extract_features(ref_frame)

    # matches = match_frames(des, des2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 32:
    #         if m.distance < 0.75*n.distance:
    #             good.append(m)

    # match_kps = []
    # ret = [] 

    # for match in good:
    #     match_kps.append((kps[match.queryIdx], kps2[match.trainIdx]))

    #     pt = kps[match.queryIdx].pt 
    #     next_pt = kps2[match.trainIdx].pt
    #     ret.append((pt, next_pt))

    #     # cv2.line(result_img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0, 255), 5)

    # match_kps = np.array(match_kps)

    # assert len(ret) >= 8
    # ret = np.array(ret)

    # print(ret[:, 0].shape[0])
    # print(ret[:, 1].shape[0])

    # for feature in ret[:, 0]:
    #     print(feature)

    # model, inliers = ransac((ret[:, 0], ret[:,1]),
    #                     EssentialMatrixTransform,
    #                     min_samples=10,
    #                     residual_threshold=RANSAC_RESIDUAL_THRES,
    #                     max_trials=RANSAC_MAX_TRIALS)

    # essential_mat, _ = cv2.findEssentialMat(good_old, good_new, camera_matrix,cv2.RANSAC)

    # points, R, t, mask = cv2.recoverPose(essential_mat, good_old, good_new)

    return match_frames(kps1, kps2, des1, des2)

if __name__=="__main__":
    image1 = cv2.imread("frames/slam_011.jpg")
    image2 = cv2.imread("frames/slam_012.jpg")

    result_image = image1.copy()

    cap = cv2.VideoCapture("videos/test_countryroad.mp4")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    F = 525

    camera_matrix = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

    R, t = getRT(image1, image2, camera_matrix)

    x = t[0]
    y = t[2]

    print(x, y)

    plt.imshow(result_image)

    plt.show()



