import cv2
import face_recognition

known_image = face_recognition.load_image_file("face_rec_3\\pic.jpg")
known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
known_face_encodings = face_recognition.face_encodings(known_image_rgb, model="large")

video_capture = cv2.VideoCapture("face_testing_codes\\video2.mp4")

face_cascade = cv2.CascadeClassifier(
    "face_testing_codes\haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error reading frame from video capture.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.04,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        cv2.putText(
            frame,
            "No faces found",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    else:
        for x, y, w, h in faces:
            face_image = frame_rgb[y : y + h, x : x + w]
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(
                face_image_rgb, model="large"
            )

            match = False
            for face_encoding in face_encodings:
                results = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                if results[0]:
                    match = True

            if match:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Harsha",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Not match",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

    cv2.imshow("Video", frame)

    if cv2.waitKey(10) & 0xFF == ord("e"):
        break

video_capture.release()
cv2.destroyAllWindows()