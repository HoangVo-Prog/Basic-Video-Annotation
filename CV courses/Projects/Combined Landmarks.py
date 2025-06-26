import cv2
import mediapipe as mp
import os
from moviepy.editor import VideoFileClip
import json

# Video input/output paths
video_url = r'D:\Python\Annotation\CV courses\Data Videos\2__EheH6kMpk3A__6.560-12.505.mp4'
video_name = os.path.basename(video_url)
output_url = r'D:\Python\Annotation\CV courses\Result\annotated_' + f'{video_name}.mp4'
final_output_url = r'D:\Python\Annotation\CV courses\new\new_' + f'{video_name}'
output_json_file = r'D:\Python\Annotation\CV courses\json\landmarks_output.json'

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print('Error opening video stream or file')
    exit()

# Get video dimensions and FPS from the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Ensure you get the correct FPS
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize MediaPipe solutions
mpHands = mp.solutions.hands
mpFaceMesh = mp.solutions.face_mesh
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Initialize video writer with the correct FPS and resolution
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_url, fourcc, fps, (width, height))

# TODO: rename, define all the path
# Initialize the data structure for JSON output
data = {
    "video_id": video_name,
    "frames": []
}

with mpHands.Hands() as hands, mpFaceMesh.FaceMesh() as faceMesh, mpPose.Pose() as pose:
    frame_idx = 0
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the frame to RGB for MediaPipe
        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame_data = {
            "frame_id": frame_idx + 1,
            "face_keypoints": [],
            "body_keypoints": [],
            "hand_keypoints": []
        }

        # Hand landmarks
        hand_results = hands.process(imgRBG)
        if hand_results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                mpDraw.draw_landmarks(
                    img, hand_landmarks, mpHands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    frame_data["hand_keypoints"].append({
                        "id": idx + 1,
                        "x": landmark.x * img.shape[1],
                        "y": landmark.y * img.shape[0],
                        "z": landmark.z,
                        "hand_index": hand_index
                    })

        # Face landmarks
        face_results = faceMesh.process(imgRBG)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mpDraw.draw_landmarks(
                    img, face_landmarks, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec
                )
                for idx, landmark in enumerate(face_landmarks.landmark):
                    frame_data["face_keypoints"].append({
                        "id": idx + 1,
                        "x": landmark.x * img.shape[1],
                        "y": landmark.y * img.shape[0],
                        "z": landmark.z,
                    })

        # Pose landmarks
        pose_results = pose.process(imgRBG)
        if pose_results.pose_landmarks:
            mpDraw.draw_landmarks(
                img, pose_results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2)
            )
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                frame_data["body_keypoints"].append({
                    "id": idx + 1,
                    "x": landmark.x * img.shape[1],
                    "y": landmark.y * img.shape[0],
                    "z": landmark.z,
                })

        # Add frame data to the output
        data["frames"].append(frame_data)

        # Write the processed frame to the output video
        out.write(img)

        # Introduce a delay to match the original FPS and keep video duration consistent
        if frame_idx < frame_count:
            # 1000 milliseconds divided by the FPS gives the correct frame duration in ms
            cv2.waitKey(int(1000 / fps))
            frame_idx += 1

        # Stop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release OpenCV objects
cap.release()
out.release()
cv2.destroyAllWindows()

# Add audio to the processed video using moviepy
original_video = VideoFileClip(video_url)
annotated_video = VideoFileClip(output_url)

# Extract audio from the original video
audio = original_video.audio

# Set the extracted audio to the annotated video
final_video = annotated_video.set_audio(audio)

# Save the final video with audio
final_video.write_videofile(final_output_url, codec='libx264', audio_codec='aac')


with open(output_json_file, 'w+', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Final video with audio saved to {final_output_url}")
