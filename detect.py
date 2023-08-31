import cv2

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open video capture (0 for webcam, "video.mp4" for a video file)
cap = cv2.VideoCapture('face.mp4')

# Dictionary to store face names
face_names = {
    0: "Harsha",
    1: "Person 2",
    # Add more names as needed
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes around faces and add names
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Set color to red
        
        # Add the name to the box if available in the dictionary
        if idx in face_names:
            name = face_names[idx]
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Set color to red
    
    # Display the frame
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
