import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(r"D:\Python\Annotation\CV courses\Data Videos\8028802-uhd_3840_2160_24fps.mp4")
pTime = 0

if not cap.isOpened():
    print('Error opening video stream or file')
    exit()

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate window dimensions (adjust as needed)
desired_width = 600
desired_height = int(desired_width * (height / width))

# Create window with calculated dimensions
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', desired_width, desired_height)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# while True:
#     success, img = cap.read()
#
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceDetection.process(imgRGB)
#     print(results)
#     if results.detections:
#         for id, detection in enumerate(results.detections):
#             mpDraw.draw_detection(img, detection)
#             # print(id, detection)
#             # print(detection.score)
#             # print(detection.location_data.relative_bounding_box)
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, ic = img.shape
#             bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),
#             int(bboxC.width * iw), int(bboxC.height * ih)
#             cv2.rectangle(img, bbox, (255, 0, 255), 2)
#             cv2.putText(img, f'{int(detection.score[0] * 100)}%',
#            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
#             2, (255, 0, 255), 2)
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
#     3, (0, 255, 0), 2)
#     cv2.imshow("Video", img)
#     cv2.waitKey(1)


with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
              mp_drawing.draw_detection(image, detection)
      # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Video', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()