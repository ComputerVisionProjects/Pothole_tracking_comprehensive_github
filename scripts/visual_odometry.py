import cv2
import numpy as np

# Load camera intrinsic parameters (fx, fy, cx, cy)
K = np.array([[718.856, 0, 607.1928],
              [0, 718.856, 185.2157],
              [0, 0, 1]])  # Example values, change based on your camera

# Initialize ORB feature detector
orb = cv2.ORB_create()

# Initialize video capture
cap = cv2.VideoCapture(1)  # Change to 0 for webcam

# Read first frame and detect features
ret, prev_frame = cap.read()
gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = orb.detectAndCompute(gray_prev, None)

# Camera pose
R_total = np.eye(3)
t_total = np.zeros((3, 1))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_prev, des_curr)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched keypoints
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute essential matrix
    E, mask = cv2.findEssentialMat(pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover pose (R and t)
    _, R, t, mask = cv2.recoverPose(E, pts_curr, pts_prev, K)
    
    # Normalize translation to avoid scale drift
    t = t / np.linalg.norm(t)
    
    # Update total pose
    t_total += R_total @ t  # Convert t to world coordinates
    R_total = R @ R_total
    
    # Display displacement
    print(f"Displacement: X={t_total[0,0]:.2f}, Y={t_total[1,0]:.2f}, Z={t_total[2,0]:.2f}")
    
    # Update previous frame
    kp_prev, des_prev = kp_curr, des_curr
    gray_prev = gray_curr
    
cap.release()
cv2.destroyAllWindows()