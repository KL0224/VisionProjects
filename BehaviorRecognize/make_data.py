import cv2
import mediapipe as mp
import pandas as pd
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Lưu giá trị khung xương
lm_list = []
label = "BODYSWING"
no_of_frames = 600

def MakeLandmarkTimestep(result):
    print(result.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(result.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)

    return c_lm

def DrawLandmarkOnImage(mpDraw, result, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(result.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), cv2.FILLED)

    return img
while len(lm_list) < no_of_frames:
    ret, frame = cap.read()
    if ret:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frameRGB)

        if result.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = MakeLandmarkTimestep(result)
            lm_list.append(lm)

            # Vẽ khung xương lên ảnh
            frame = DrawLandmarkOnImage(mpDraw, result, frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Viết vào file csv
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()