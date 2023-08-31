# import cv2
# import numpy as np

# # Open video capture (0 for webcam, "video.mp4" for a video file)
# cap = cv2.VideoCapture('firec.mp4')

# # Define the desired smaller frame size
# small_frame_size = (640, 480)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Resize the frame to the smaller frame size
#     small_frame = cv2.resize(frame, small_frame_size)
    
#     # Convert frame to HSV color space
#     hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
#     # Define lower and upper bounds for fire color in HSV
#     lower_bound = np.array([0, 120, 70])
#     upper_bound = np.array([20, 255, 255])
    
#     # Create a mask to detect fire color
#     mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    
#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Draw bounding boxes around fire-like regions
#     for contour in contours:
#         if cv2.contourArea(contour) > 200:  # Adjust this threshold as needed
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Draw red bounding box
    
#     # Display the smaller frame
#     cv2.imshow('Fire Detection', small_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Open video capture (0 for webcam, "video.mp4" for a video file)
cap = cv2.VideoCapture('firec.mp4')

# Define the desired smaller frame size
small_frame_size = (640, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to the smaller frame size
    small_frame = cv2.resize(frame, small_frame_size)
    
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for fire color in HSV
    lower_bound = np.array([0, 120, 70])
    upper_bound = np.array([20, 255, 255])
    
    # Create a mask to detect fire color
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any fire-like regions are detected
    fire_alert = False
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Draw red bounding box
            fire_alert = True
    
    # Display fire alert text if fire is detected
    if fire_alert:
        cv2.putText(small_frame, "Fire Alert!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the smaller frame
    cv2.imshow('Fire Detection', small_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
