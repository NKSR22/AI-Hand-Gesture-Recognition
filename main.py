import cv2
import mediapipe as mp
import time
import numpy as np
import math
import csv
import os

# =============================================================================
# การตั้งค่าเบื้องต้น (Configuration)
# =============================================================================
DATA_FILE = "hand_data.csv"   # ชื่อไฟล์สำหรับเก็บข้อมูล Training
K_NEIGHBORS = 3               # จำนวนเพื่อนบ้านที่ใกล้ที่สุดสำหรับ KNN Algorithm

# การตั้งค่า MediaPipe (ไลบรารีสำหรับตรวจจับมือ)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ตัวแปร Global (ใช้ร่วมกันทั้งโปรแกรม)
detection_result = None       # เก็บผลลัพธ์การตรวจจับล่าสุด
training_data = []            # ข้อมูล Features ที่ใช้สอน AI
training_labels = []          # คำตอบ (Label) ของข้อมูลนั้นๆ (เช่น 0, 1, 2...)
should_quit = False           # ตัวแปรควบคุมการปิดโปรแกรม

# =============================================================================
# ฟังก์ชันจัดการข้อมูล Training (Data Management)
# =============================================================================

def load_training_data():
    """
    โหลดข้อมูล Training จากไฟล์ CSV เข้าสู่หน่วยความจำ
    Load training data from CSV file into memory.
    """
    global training_data, training_labels
    training_data.clear()
    training_labels.clear()
    
    if not os.path.exists(DATA_FILE):
        return

    try:
        with open(DATA_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # คอลัมน์แรกคือ Label (คำตอบ), ส่วนที่เหลือคือ Features (ข้อมูลจุด)
                label = int(row[0])
                features = np.array([float(x) for x in row[1:]], dtype=np.float32)
                training_data.append(features)
                training_labels.append(label)
        print(f"Loaded {len(training_data)} training samples. (โหลดข้อมูลแล้ว {len(training_data)} ตัวอย่าง)")
    except Exception as e:
        print(f"Error loading data: {e}")

def save_sample(features, label):
    """
    บันทึกข้อมูลตัวอย่างใหม่ลงไฟล์ CSV และหน่วยความจำ
    Save a new training sample to CSV and memory.
    """
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([label] + features.tolist())
    
    training_data.append(features)
    training_labels.append(label)
    print(f"Saved sample for number: {label} (บันทึกท่าทางสำหรับเลข {label})")

# =============================================================================
# ฟังก์ชันประมวลผลทางคณิตศาสตร์ (Math & ML Functions)
# =============================================================================

def process_landmarks(hand_landmarks):
    """
    แปลงข้อมูลจุด 21 จุด ให้เป็น Feature Vector (ตัวเลขชุดเดียว) เพื่อให้ AI เข้าใจง่าย
    Converts 21 landmarks into a normalized feature vector.
    
    หลักการ:
    1. แปลงเป็น NumPy Array
    2. ย้ายจุดศูนย์กลางไปที่ข้อมือ (Translation Invariance)
    3. ปรับขนาดให้เท่ากัน (Scale Invariance) เพื่อให้มือใกล้/ไกล มีค่าเท่ากัน
    """
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks])
    
    # ย้ายจุดศูนย์กลาง (Relative to wrist)
    wrist = coords[0]
    coords = coords - wrist
    
    # ปรับขนาด (Normalize)
    max_value = np.max(np.abs(coords))
    if max_value > 0:
        coords = coords / max_value
        
    # แปลงเป็น Vector 1 มิติ
    return coords.flatten().astype(np.float32)

def predict_knn(features):
    """
    ทำนายผลด้วย KNN (K-Nearest Neighbors)
    เปรียบเทียบข้อมูลปัจจุบัน กับข้อมูลที่เคยสอนไว้ที่ 'เหมือนที่สุด'
    Predicts the gesture using KNN algorithm.
    """
    if not training_data: return None, 0.0
    
    # คำนวณระยะห่าง (Euclidean Distance) กับข้อมูลทั้งหมด
    data_matrix = np.array(training_data)
    diff = data_matrix - features
    dists = np.linalg.norm(diff, axis=1)
    
    # หา K ตัวที่ใกล้ที่สุด
    k = min(K_NEIGHBORS, len(training_data))
    nearest_indices = dists.argsort()[:k]
    nearest_labels = [training_labels[i] for i in nearest_indices]
    
    # โหวตหาคำตอบที่มีมากที่สุด (Majority Vote)
    counts = np.bincount(nearest_labels)
    prediction = np.argmax(counts)
    
    # คำนวณความมั่นใจ (Confidence)
    confidence = counts[prediction] / k
    return prediction, confidence

def calculate_distance(p1, p2):
    """คำนวณระยะห่างระหว่างจุด 2 จุด (Euclidean Distance)"""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_fingers_heuristic(hand_landmarks, handedness):
    """
    ฟังก์ชันนับนิ้วแบบดั้งเดิม (ใช้ตรรกะเรขาคณิต)
    Standard finger counting logic using geometry (Distance check).
    """
    fingers = []
    
    wrist = hand_landmarks[0]      # ข้อมือ
    index_mcp = hand_landmarks[5]  # โคนนิ้วชี้
    thumb_tip = hand_landmarks[4]  # ปลายนิ้วโป้ง
    thumb_ip = hand_landmarks[3]   # ข้อต่อนิ้วโป้ง
    
    # 1. เช็คนิ้วโป้ง (Thumb) - ดูเทียบกับโคนนิ้วชี้
    dist_tip_to_index = calculate_distance(thumb_tip, index_mcp)
    dist_ip_to_index = calculate_distance(thumb_ip, index_mcp)
    
    if dist_tip_to_index > dist_ip_to_index: 
        fingers.append(1) # เปิด (Open)
    else: 
        fingers.append(0) # ปิด (Closed)

    # 2. เช็ค 4 นิ้วที่เหลือ (Fingers) - ดูระยะห่างจากข้อมือ
    finger_tips = [8, 12, 16, 20] # ปลายนิ้ว
    finger_pips = [6, 10, 14, 18] # ข้อต่อนิ้ว (Knuckles)
    
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        # ถ้าปลายนิ้วอยู่ไกลจากข้อมือ มากกว่าข้อต่อ = นิ้วกางออก
        if calculate_distance(wrist, hand_landmarks[tip_id]) > calculate_distance(wrist, hand_landmarks[pip_id]):
            fingers.append(1)
        else: 
            fingers.append(0)
            
    return fingers.count(1)

# =============================================================================
# ส่วนแสดงผลและจัดการ UI (Visualization & UI)
# =============================================================================

def print_result(result, output_image, timestamp_ms):
    """Callback function รับค่าจาก MediaPipe แบบ Asynchronous"""
    global detection_result
    detection_result = result

def draw_landmarks_on_image(rgb_image, detection_result):
    """วาดเส้นและจุดบนมือ (Draw skeleton on hand)"""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    if not hand_landmarks_list: return annotated_image
    
    height, width, _ = annotated_image.shape
    
    # คู่จุดที่ต้องลากเส้นเชื่อมกัน (Skeleton connections)
    CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
                   (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17),(5,9),(9,13),(13,17)]

    for hand_landmarks in hand_landmarks_list:
        # วาดเส้น (Lines)
        for start_idx, end_idx in CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            cv2.line(annotated_image, (int(start.x*width), int(start.y*height)), 
                     (int(end.x*width), int(end.y*height)), (0, 255, 0), 2)
        # วาดจุด (Points)
        for lm in hand_landmarks:
            cv2.circle(annotated_image, (int(lm.x*width), int(lm.y*height)), 5, (0, 0, 255), -1)
            
    return annotated_image

def mouse_callback(event, x, y, flags, param):
    """ฟังก์ชันดักจับการคลิกเมาส์ (Handle Mouse Click)"""
    global should_quit
    if event == cv2.EVENT_LBUTTONDOWN:
        # ตรวจสอบว่าคลิกโดนปุ่ม EXIT มุมขวาบนหรือไม่
        width = param
        # พื้นที่ปุ่ม: x ตั้งแต่ (width-100) ถึง (width-10), y ตั้งแต่ 10 ถึง 60
        if width - 100 <= x <= width - 10 and 10 <= y <= 60:
            should_quit = True

# =============================================================================
# ฟังก์ชันหลัก (Main Execution)
# =============================================================================

def main():
    global detection_result, should_quit
    
    # 1. โหลดข้อมูล Training
    load_training_data()
    
    # 2. ตั้งค่า MediaPipe Hand Landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,                         # ตรวจจับทีละ 1 มือ (เพื่อการสอนที่ง่าย)
        min_hand_detection_confidence=0.7,   # ค่าความมั่นใจขั้นต่ำในการตรวจจับ (70%)
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        result_callback=print_result)

    # 3. เปิดกล้อง (Camera Initialization)
    cap = None
    for i in range(2): # ลองวนหา camera index 0 และ 1
        temp = cv2.VideoCapture(i)
        if temp.isOpened(): cap = temp; break
    
    if not cap: 
        print("No camera found (ไม่พบกล้อง Webcam)")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # สร้างหน้าต่างโปรแกรมและผูก Mouse Callback
    window_name = 'Teachable Hand Tracker - AI Lab'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=frame_width)

    print("AI Hand System Running. Click 'EXIT' or Press 'Q' to quit.")

    # 4. เริ่มลูปการทำงาน (Main Loop)
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            if should_quit: break # ตรวจสอบปุ่ม Exit
            
            success, image = cap.read()
            if not success: break
            
            # กลับภาพซ้าย-ขวา (Mirror) เพื่อให้เหมือนส่องกระจก
            image = cv2.flip(image, 1)
            
            # MediaPipe ต้องการภาพแบบ RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # ส่งภาพไปประมวลผลแบบ Asynchronous
            landmarker.detect_async(mp_image, int(time.time() * 1000))
            
            # --- ส่วนประมวลผล Logic (Processing) ---
            current_result = detection_result
            display_image = image.copy()
            
            prediction_text = "Waiting..."
            mode_text = "Heuristic (Default)"
            color = (200, 200, 200)
            
            if current_result and current_result.hand_landmarks:
                # วาด Skeleton
                display_image = draw_landmarks_on_image(display_image, current_result)
                
                hand_lms = current_result.hand_landmarks[0]
                handedness = current_result.handedness[0]
                features = process_landmarks(hand_lms)
                
                # ตัดสินใจว่าจะใช้วิธีไหนนับนิ้ว?
                if len(training_data) > 0:
                    # ถ้ามีข้อมูลสอน -> ใช้ AI (KNN) จำแนกท่าทาง
                    label, conf = predict_knn(features)
                    prediction_text = f"AI Custom: {label}"
                    mode_text = f"Learned Model ({len(training_data)} samples)"
                    color = (0, 255, 0) # สีเขียว (AI)
                else:
                    # ถ้าไม่มีข้อมูลสอน -> ใช้ Logic ปกติ
                    count = count_fingers_heuristic(hand_lms, handedness)
                    prediction_text = f"Count: {count}"
                    mode_text = "Basic Logic"
                    color = (0, 255, 255) # สีเหลือง (Logic)

                # รับค่าปุ่มกด (Keyboard Input)
                key = cv2.waitKey(1) & 0xFF
                if ord('0') <= key <= ord('9'):        # กด 0-9 เพื่อสอน AI
                    label = key - ord('0')
                    save_sample(features, label)
                    cv2.rectangle(display_image, (0,0), (display_image.shape[1], display_image.shape[0]), (255,255,255), cv2.FILLED) # Flash Effect
                elif key == ord('c'):                  # กด C เพื่อล้างข้อมูล
                    training_data.clear()
                    training_labels.clear()
                    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
                    print("Training data cleared!")
                elif key == ord('q'): should_quit = True
            
            # --- ส่วนแสดงผลบนหน้าจอ (UI Overlays) ---
            # 1. กรอบแสดงผลลัพธ์ (Info Box)
            cv2.rectangle(display_image, (0, 0), (450, 100), (0, 0, 0), cv2.FILLED)
            cv2.putText(display_image, prediction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(display_image, mode_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # คำแนะนำด้านล่าง
            cv2.putText(display_image, "Press 0-5 to TRAIN | C to Clear", (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
            
            # 2. ปุ่ม EXIT มุมขวาบน
            btn_x1, btn_y1 = frame_width - 100, 10
            btn_x2, btn_y2 = frame_width - 10, 60
            cv2.rectangle(display_image, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 200), cv2.FILLED) # กล่องสีแดง
            cv2.putText(display_image, "EXIT", (btn_x1 + 15, btn_y1 + 35), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

            cv2.imshow(window_name, display_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    # ปิดการทำงานและคืนทรัพยากร
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
