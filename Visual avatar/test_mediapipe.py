import mediapipe as mp
import cv2
import time


video_name = "720p"
video_url = f'{video_name}.mp4'
cap = cv2.VideoCapture(video_url)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
pTime = 0
print(size)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
# drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(f'new_{video_name}.mp4', fourcc, 20.0, size)

while True:
    success, img = cap.read()
    if not success:
        print("Error reading video frame. Exiting...")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)
            # for id, lm in enumerate(faceLms.landmark):
            #     # print(lm)
            #     ih, iw, ic = img.shape
            #     x, y = int(lm.x * iw), int(lm.y * ih)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                1, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    out.write(img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
