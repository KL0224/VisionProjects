import cv2
import mediapipe as mp
import pandas as pd
import keras
import numpy as np
import threading

# Đọc ảnh từ webcam
cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load mode
model = keras.models.load_model('lstm.h5')

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

def DrawClassOnImage(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def Detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "SWING BODY"
    else:
        label = "SWING HAND"
    return label

# Lưu giá trị khung xương
lm_list = []
label = "Warmup...."
no_of_timesteps = 10
i = 0
warmup_frames = 60

while True:
    ret, frame = cap.read()
    if ret:
        i = i + 1
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frameRGB)

        if i > warmup_frames:
            print("Start Detection")
            if result.pose_landmarks:
                # Ghi nhận thông số khung xương
                lm = MakeLandmarkTimestep(result)
                lm_list.append(lm)
                if len(lm_list) == no_of_timesteps:
                    # Đưa vào model nhận diện
                    thread_01 = threading.Thread(target=Detect, args=(model, lm_list))
                    thread_01.start()
                    lm_list = []

                # Vẽ khung xương lên ảnh
                frame = DrawLandmarkOnImage(mpDraw, result, frame)

        frame = DrawClassOnImage(label, frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Viết vào file csv
cap.release()
cv2.destroyAllWindows()