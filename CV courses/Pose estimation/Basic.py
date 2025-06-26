import cv2
import mediapipe as mp
import time

video_path = r'D:\Python\Annotation\CV courses\Data Videos\7884018-uhd_4096_2160_25fps.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Error opening video stream or file')
    exit()

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate window dimensions (adjust as needed)
desired_width = 800
desired_height = int(desired_width * (height / width))

# Create window with calculated dimensions
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', desired_width, desired_height)

pTime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (100, 200), cv2.FONT_HERSHEY_PLAIN, 15,
                (255, 0, 0), 3)
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
