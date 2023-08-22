import cv2
import winsound

video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

danger_line_start = (800, 550)
danger_line_end = (1000, 470)

person_in_danger = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1300, 780))
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    persons = person_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in persons:
        person_center = (x + w // 2, y + h // 2)
        if (danger_line_start[1] - person_center[1]) * (danger_line_end[0] - person_center[0]) - (danger_line_start[0] - person_center[0]) * (danger_line_end[1] - person_center[1]) > 0:
            person_in_danger = True
            color = (0, 0, 255)
            
            # Play a beep sound when a person crosses the line
            winsound.Beep(1000, 200)  # Frequency: 1000 Hz, Duration: 200 ms
        else:
            person_in_danger = False
            color = (0, 255, 0) 
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    cv2.line(frame, danger_line_start, danger_line_end, (0, 0, 255), 2)
    
    cv2.imshow("Video Analysis", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()