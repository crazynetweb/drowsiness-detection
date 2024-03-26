import cv2
import time

# Define eye aspect ratio (EAR) threshold
EAR_THRESH = 0.2

# Define consecutive frames for which EAR is below threshold to trigger alert
ALERT_FRAMES_THRESH = 50

# Initialize counters
consecutive_frames = 0
last_eye_status = False  # Assuming eyes are open initially

# Load pre-trained facial landmark detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Open video capture (change 0 to video path for file)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2)

    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Process each eye
        for (ex, ey, ew, eh) in eyes:
            # Calculate eye aspect ratio (EAR)
            eye_center = (ex + (ew // 2), ey + (eh // 2))
            left_eye_start = (ex, ey)
            right_eye_end = (ex + ew, ey + eh)
            cv2.line(roi_color, left_eye_start, eye_center, (0, 0, 255), 2)
            cv2.line(roi_color, right_eye_end, eye_center, (0, 0, 255), 2)
            vertical_distance = (ey + eh) - ey
            horizontal_distance = (ex + ew) - ex
            EAR = vertical_distance / (horizontal_distance * 2)

            # Update eye status based on EAR
            eye_status = EAR > EAR_THRESH

            # Check for consecutive frames with low EAR
            if not eye_status:
                consecutive_frames += 1
            else:
                consecutive_frames = 0

            # Trigger drowsiness alert
            if consecutive_frames >= ALERT_FRAMES_THRESH:
                cv2.putText(frame, "Drowsiness Alert!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Add sound or vibration alert here (not included in this example)
                last_eye_status = True  # Set flag to avoid multiple consecutive alerts
                time.sleep(1)  # Delay next alert for 1 second

    # Display frame with detections (optional)
    cv2.imshow('Drowsiness Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()