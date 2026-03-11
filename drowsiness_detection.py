import cv2
import mediapipe as mp

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

EAR_THRESH = 0.25
MAR_THRESH = 0.6
DROWSY_FRAMES = 40
YAWN_FRAMES = 15

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]

MOUTH_UPPER = [13, 312, 82, 13]
MOUTH_LOWER = [14, 317, 87, 14]
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

counter = 0
yawn_counter = 0

def euclidean_dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def eye_aspect_ratio(landmarks, indices, img_w, img_h):
    points = [(int(landmarks.landmark[i].x * img_w), int(landmarks.landmark[i].y * img_h)) for i in indices]
    A = euclidean_dist(points[1], points[5])
    B = euclidean_dist(points[2], points[4])
    C = euclidean_dist(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def improved_mouth_aspect_ratio(landmarks, img_w, img_h):
    points_upper = [(int(landmarks.landmark[i].x * img_w), int(landmarks.landmark[i].y * img_h)) for i in MOUTH_UPPER]
    points_lower = [(int(landmarks.landmark[i].x * img_w), int(landmarks.landmark[i].y * img_h)) for i in MOUTH_LOWER]
    verticals = [euclidean_dist(u, l) for u, l in zip(points_upper, points_lower)]
    vertical_avg = sum(verticals) / len(verticals)
    
    left = (int(landmarks.landmark[MOUTH_LEFT].x * img_w), int(landmarks.landmark[MOUTH_LEFT].y * img_h))
    right = (int(landmarks.landmark[MOUTH_RIGHT].x * img_w), int(landmarks.landmark[MOUTH_RIGHT].y * img_h))
    horizontal = euclidean_dist(left, right)
    
    mar = vertical_avg / horizontal
    return mar

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0]

        left_ear = eye_aspect_ratio(mesh, LEFT_EYE_INDICES, w, h)
        right_ear = eye_aspect_ratio(mesh, RIGHT_EYE_INDICES, w, h)
        ear = (left_ear + right_ear) / 2.0

        mar = improved_mouth_aspect_ratio(mesh, w, h)

        # Check eye closure (drowsiness)
        if ear < EAR_THRESH:
            counter += 1
            if counter >= DROWSY_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            counter = 0

        # Check yawning
        if mar > MAR_THRESH:
            yawn_counter += 1
            cv2.putText(frame, "YAWNING!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
        else:
            yawn_counter = 0

        # Display EAR and MAR for debugging
        cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Driver Sleepiness and Yawning Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
