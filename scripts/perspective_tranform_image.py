import cv2

# Load the original image
image_path = "D:\Arybhatta_motors_computer_vision\Pothole-tracking-comprehensive\scripts\pptrnsfrm_images_lab\scene00135.png"
img = cv2.imread(image_path)

if img is None:
    print("Could not load image.")
    exit()

# Resize to 1280x720 (720p) for display
resized_img = cv2.resize(img, (1280, 720))
clone = resized_img.copy()
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        points.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image (720p)", clone)

# Show the resized image and set the callback
cv2.imshow("Image (720p)", clone)
cv2.setMouseCallback("Image (720p)", click_event)

print("Click on the image (720p). Press 'q' to quit.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Final list of 720p coordinates
print("Points clicked (720p):", points)
