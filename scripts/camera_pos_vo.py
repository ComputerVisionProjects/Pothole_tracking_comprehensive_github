import cv2
import numpy as np

# Camera intrinsic matrix (K) - Adjust based on your camera calibration
K = np.array([[892.86025081, 0, 319.72816385],
                   [0, 894.36852612, 182.8589494 ],
                      [0, 0, 1]])

# Capture video (adjust index or file path as needed)
cap = cv2.VideoCapture(1)  # Use "video.mp4" instead of 0 for a file

# Initialize ORB detector
orb = cv2.ORB_create(3000)

# Initialize previous frame data
ret, frame_prev = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = orb.detectAndCompute(gray_prev, None)

# Camera position in 3D space (initial)
camera_position = np.zeros((3, 1))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    if not kp_prev or not kp or des_prev is None or des is None:
        print("No keypoints detected. Skipping frame.")
        frame_prev, kp_prev, des_prev = frame, kp, des
        continue

    # Match keypoints using Brute Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_prev, des)
    matches = sorted(matches, key=lambda x: x.distance)

    # Ensure valid matches exist
    valid_matches = [m for m in matches if 0 <= m.queryIdx < len(kp_prev) and 0 <= m.trainIdx < len(kp)]
    if len(valid_matches) < 10:
        print("Not enough valid matches. Skipping frame.")
        frame_prev, kp_prev, des_prev = frame, kp, des
        continue

    # Extract matched keypoints
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in valid_matches])
    pts_curr = np.float32([kp[m.trainIdx].pt for m in valid_matches])

    # Estimate Essential Matrix
    E, _ = cv2.findEssentialMat(pts_curr, pts_prev, K, method=cv2.RANSAC, threshold=1.0)

    if E is not None and E.shape == (3, 3):
        # Recover camera movement (R = Rotation, t = Translation)
        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, K)

        # Update camera position (Z increases forward)
        camera_position += R @ t

        # Print camera position
        print(f"Camera Position: X={camera_position[0,0]:.2f}, Y={camera_position[1,0]:.2f}, Z={camera_position[2,0]:.2f}")

    # Draw matches
    match_img = cv2.drawMatches(frame_prev, kp_prev, frame, kp, valid_matches[:min(50, len(valid_matches))], None)
    cv2.imshow("Visual Odometry", match_img)

    # Update previous frame data
    frame_prev, kp_prev, des_prev = frame, kp, des

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
