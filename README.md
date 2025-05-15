# 📌 Vision Projects

## 1. Tổng quan

Dự án bao gồm một số project sử dụng các framework và thư viện như:

- **Frameworks**: OpenCV, YOLO, TensorFlow
- **Thư viện hỗ trợ**: NumPy, Mediapipe, Hand Detection Library...

Nhằm giải quyết các bài toán thị giác máy tính như:

- Nhận diện hành động
- Trò chơi Đấm - Lá - Kéo bằng camera
- Vẽ sơ đồ **heatmap** thể hiện mật độ di chuyển của khách hàng trong cửa hàng

---

## 2. Thông tin chi tiết các project

### 2.1. BehaviorRecognize – Nhận diện hành vi con người

- **Mục tiêu**: Nhận diện hành vi của con người thông qua camera.
- **Công nghệ sử dụng**:
  - `OpenCV`: Đọc các khung hình (frame) từ webcam.
  - `Mediapipe`: Nhận diện khung xương chuyển động (pose estimation).
  - `LSTM`: Dự đoán hành động dựa trên chuỗi khung hình đầu vào.
- **Quy trình hoạt động**:
  1. Thu thập dữ liệu video từ webcam.
  2. Trích xuất các khung xương người từ khung hình (skeleton data).
  3. Dự đoán hành động sử dụng mạng LSTM.
  4. Hiển thị hành động dự đoán được trên khung ảnh video đầu ra.
- **Dữ liệu huấn luyện**:
  - Hai hành động: `handswing` và `bodyswing`
  - Mỗi hành động bao gồm **600 frame ảnh**.

---

### 2.2. DamLaKeo – Trò chơi Đấm - Lá - Kéo

- **Mục tiêu**: Hiện thực trò chơi kéo-búa-bao sử dụng camera và nhận diện cử chỉ tay.
- **Công nghệ sử dụng**:
  - `OpenCV`: Truy xuất dữ liệu từ camera.
  - `hand_detection_lib`: Nhận diện bàn tay người dùng.
- **Quy trình hoạt động**:
  1. Phát hiện cử chỉ tay (kéo, búa, bao).
  2. Máy tính ngẫu nhiên chọn một trong ba hành động.
  3. So sánh kết quả giữa người chơi và máy để xác định người thắng cuộc.
- **Giao diện**: Hiển thị trực tiếp kết quả trên màn hình với thông tin người chơi, hành động và kết quả.

---

### 2.3. StoreHeatMap – Bản đồ nhiệt (Heatmap) trong cửa hàng

- **Mục tiêu**: Phân tích video giám sát để tạo bản đồ nhiệt thể hiện mật độ khách hàng.
- **Công nghệ sử dụng**:
  - `YOLOv4`: Phát hiện người trong video.
- **Quy trình hoạt động**:
  1. Phân tích video để phát hiện vị trí người tại từng khung hình.
  2. Ghi nhận số lần xuất hiện tại các vị trí.
  3. Vẽ heatmap lên video để trực quan hóa khu vực có lượng người qua lại nhiều.
- **Kết quả**: Các khu vực đông người sẽ hiện rõ ràng hơn (đậm màu), giúp hỗ trợ phân tích hành vi khách hàng hoặc tối ưu bố trí cửa hàng.
