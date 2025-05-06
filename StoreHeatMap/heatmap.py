# In Store Heatmap by Yolov4
import cv2
import numpy as np
from imutils.video import VideoStream # Tốc độ cao hơn cv2
from skimage.transform import resize

# Path
video_path = "video.mp4"
classnames_file_path = "classnames.txt"
weights_file = "yolov4-tiny.weights"
config_file = "yolov4-tiny.cfg"

# Threshold
conf_threshold = 0.5
nms_threshold = 0.4

# Detect class
detect_class = "person"

# Frame
frame_width = 1280
frame_height = 720
scale = 0.00392

# Create gird
cell_size = 40 # 40x40 pixel
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size
heat_matrix = np.zeros((n_rows, n_cols))
alpha = 0.4

# Network
yolo_net = cv2.dnn.readNet(weights_file, config_file)

# Read class names
classes = None
with open(classnames_file_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Color random
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Function return row and column from x, y
def GetRowCol(x, y):
    row = y // cell_size
    col = x // cell_size
    return row, col

# Draw Gird
def DrawGird(img):
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        color = (255, 255, 255)
        thickness = 1
        img = cv2.line(img, start_point, end_point, color, thickness)

    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        color = (255, 255, 255)
        thickness = 1
        img = cv2.line(img, start_point, end_point, color, thickness)

    return img

# Function return Output Layer
def GetOutputLayers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Function draw rectangle and class name
def DrawPredict(img, class_id, x, y, x_plus_w, y_plus_h):
    global heat_matrix
    r, c = GetRowCol((x_plus_w + x) // 2, (y_plus_h + y) // 2) # Center of person
    heat_matrix[r, c] += 1

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show
video = VideoStream(src=video_path).start()
while True:
    frame = video.read()

    # Detect item in frame
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(GetOutputLayers(yolo_net))

    # Filter object in frame
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence > conf_threshold) and (classes[class_id] == detect_class):
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw rectangle around items
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        DrawPredict(frame, class_ids[i], round(x), round(y), round(x + w), round(y + h))


    temp_heat_matrix = heat_matrix.copy() # Draw rely on temp_heat_matrix

    # Normalize
    temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))
    temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
    temp_heat_matrix = np.uint8(temp_heat_matrix * 255)

    # Create heat map
    image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

    frame = DrawGird(frame)
    # Chồng hình
    cv2.addWeighted(image_heat, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("Video frame", frame)
    #cv2.imshow("Heatmap", image_heat)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()