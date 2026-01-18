# AI Hand Tracking & Gesture Recognition (Teachable Machine)
**Real-time Hand Gesture Recognition System using MediaPipe & KNN Classifier**

---

### 🌐 Project Information
**Subject:** Introduction to Artificial Intelligence (รายวิชาปัญญาประดิษฐ์เบื้องต้น)  
**Program:** B.Ind.Tech in Electrical Technology and Intelligence Control Systems (หลักสูตร อส.บ.เทคโนโลยีไฟฟ้าและระบบควบคุมอัจฉริยะ)  
**Department:** Electrical Engineering, Faculty of Industry and Technology  
**University:** Rajamangala University of Technology Isan, Sakon Nakhon Campus (มหาวิทยาลัยเทคโนโลยีราชมงคลอีสาน วิทยาเขตสกลนคร)

**Developer / Instructor:** Nakarin Sripanya (อ.นครินทร์ ศรีปัญญา)  
**Email:** nakatin.sr@rmuti.ac.th  
**GitHub:** [https://github.com/NKSR22/AI-Hand-Gesture-Recognition](https://github.com/NKSR22/AI-Hand-Gesture-Recognition)

---

## 📖 About the Project (เกี่ยวกับโปรเจกต์)
**[EN]** This project is a real-time hand gesture recognition system developed using Python, MediaPipe, and OpenCV. It features a **"Teachable Machine"** capability, allowing users to train the AI to recognize custom hand gestures instantly using a K-Nearest Neighbors (KNN) algorithm without retraining the entire model.

**[TH]** โปรเจกต์นี้เป็นระบบตรวจจับมือและนับนิ้วแบบเรียลไทม์ พัฒนาด้วย Python, MediaPipe และ OpenCV จุดเด่นคือความสามารถแบบ **"Teachable Machine"** ที่เปิดโอกาสให้ผู้ใช้สามารถ "สอน" AI ให้จดจำท่าทางมือใหม่ๆ ได้ทันทีผ่านอัลกอริทึม KNN โดยไม่ต้องเทรนโมเดลใหม่ตั้งแต่ต้น

---

## 🧠 Theory & Principles (ทฤษฎีและหลักการทำงาน)

### 1. Computer Vision & Landmark Detection
**[EN]** The system utilizes **MediaPipe Hands**, a high-fidelity hand tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from a single video frame.
- **Palm Detection Model:** Operates on the full image and returns an oriented hand bounding box.
- **Hand Landmark Model:** Operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints.

**[TH]** ระบบใช้ **MediaPipe Hands** ซึ่งเป็นโซลูชันการติดตามมือที่มีความแม่นยำสูง โดยใช้ Machine Learning ในการคาดการณ์จุดพิกัด 3 มิติ (3D Landmarks) จำนวน 21 จุดบนมือจากภาพวิดีโอ
- **โมเดลตรวจจับฝ่ามือ:** ค้นหาตำแหน่งมือในภาพรวม
- **โมเดลหาจุดพิกัดมือ:** ทำงานในกรอบภาพที่ตัดเฉพาะส่วนมือ เพื่อหาตำแหน่งข้อต่อต่างๆ อย่างละเอียด

![Hand Landmarks](https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png)

### 2. Feature Extraction (การสกัดคุณลักษณะ)
**[EN]** Raw coordinates (x, y) from the camera are not suitable for direct classification due to position and scale variations. The system performs preprocessing:
- **Translation Invariance:** All points are shifted relative to the Wrist (Point 0) so that the hand's position on the screen doesn't affect recognition.
- **Scale Invariance:** Coordinates are normalized by the hand's size to ensure consistent recognition regardless of distance from the camera.

**[TH]** พิกัดดิบ (x, y) จากกล้องไม่เหมาะแก่การนำไปจำแนกโดยตรงเนื่องจากความแปรปรวนของตำแหน่งและขนาด ระบบจึงทำการประมวลผลเบื้องต้น:
- **ไม่ขึ้นกับตำแหน่ง (Translation Invariance):** ย้ายจุดอ้างอิงทั้งหมดไปที่ข้อมือ (จุดที่ 0) เพื่อให้ตำแหน่งมือบนหน้าจอไม่มีผลต่อการจำแนก
- **ไม่ขึ้นกับขนาด (Scale Invariance):** ปรับขนาดพิกัดเทียบกับขนาดฝ่ามือ เพื่อให้สามารถจำแนกท่าทางได้ไม่ว่ามือจะอยู่ใกล้หรือไกลกล้อง

### 3. K-Nearest Neighbors (KNN) Classification
**[EN]** For the "Teachable" feature, the system uses the **K-Nearest Neighbors (KNN)** algorithm. It is a non-parametric, lazy learning algorithm.
- When a user saves a gesture, the extracted feature vector is stored in memory.
- During inference, the system calculates the **Euclidean Distance** between the current hand pose and all stored samples.
- It selects the `K` closest samples (Neighbors) and assigns the class label based on a majority vote.

**[TH]** สำหรับฟีเจอร์ "สอนได้" ระบบใช้อัลกอริทึม **K-Nearest Neighbors (KNN)** ซึ่งเป็นการเรียนรู้แบบ Lazy Learning
- เมื่อผู้ใช้บันทึกท่าทาง ข้อมูลเวกเตอร์คุณลักษณะ (Feature Vector) จะถูกเก็บในหน่วยความจำ
- ในขั้นตอนการทำนาย ระบบจะคำนวณ **ระยะห่างยูคลิด (Euclidean Distance)** ระหว่างท่ามือปัจจุบันกับข้อมูลตัวอย่างทั้งหมดที่เก็บไว้
- ระบบจะเลือกตัวอย่างที่ใกล้เคียงที่สุดจำนวน `K` ตัว (Neighbors) และตัดสินใจเลือกคำตอบจากเสียงข้างมาก (Majority Vote)

---

## 🛠 Installation & Usage (การติดตั้งและการใช้งาน)

### Prerequisites (สิ่งที่ต้องมี)
- Python 3.9+
- Webcam

### Steps (ขั้นตอน)
1. **Clone Repository:**
   ```bash
   git clone https://github.com/NKSR22/AI-Hand-Gesture-Recognition.git
   cd AI_Count_figer
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

### 🎮 How to Use (วิธีใช้งาน)
- **Heuristic Mode (Default):** The system counts fingers using standard geometry logic (Yellow status).
- **Training Mode:**
  1. Pose your hand in front of the camera.
  2. Press a number key **(0-9)** to label that pose.
  3. The system switches to **AI Mode** (Green status) and starts recognizing your custom gestures.
- **Clear Data:** Press **'C'** to reset training data.
- **Exit:** Click the **EXIT button** or press **'Q'**.

---

## 🐳 Docker Guide (คู่มือการใช้งานผ่าน Docker)
**[EN]** Running GUI applications with Webcam access in Docker requires **X11 Forwarding** configuration.
**[TH]** การรันโปรแกรมที่มี GUI และ Webcam ผ่าน Docker จำเป็นต้องตั้งค่า X11 Forwarding เพื่อให้ Container สามารถแสดงผลหน้าต่างโปรแกรมบนเครื่องของเราได้

### Step 1: Build Image
สร้าง Docker Image จาก Dockerfile (ทำเหมือนกันทุก OS)
```bash
docker build -t ai-hand-tracker .
```

### Step 2: Prepare & Run Container (OS Specific Setup)

#### 🪟 Windows (WSL2)
**Requirement:** [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/) & [usbipd-win](https://github.com/dorssel/usbipd-win)
1. **Setup XLaunch (VcXsrv):**
   - เปิดโปรแกรม **XLaunch**
   - **Display settings:** เลือก `Multiple windows`
   - **Client startup:** เลือก `Start no client`
   - **Extra settings:** **ติ๊กถูก** ที่ `Disable access control` (สำคัญมาก!)

2. **Mount Webcam (เชื่อมต่อกล้องเข้า WSL):**
   - เปิด PowerShell (Admin) แล้วรัน `usbipd list` เพื่อดู BusID ของกล้อง
   - รันคำสั่ง: `usbipd wsl attach --busid <BUSID> --distribution Ubuntu` (เปลี่ยน Ubuntu เป็นชื่อ Distro ของคุณ)

3. **Run Command (ใน WSL Terminal):**
   ```bash
   # ตั้งค่า IP ของ Display
   export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0
   
   # รัน Container
   docker run -it --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ai-hand-tracker
   ```

#### 🍎 macOS (Intel / Apple Silicon)
**Requirement:** [XQuartz](https://www.xquartz.org/)
1. **Setup XQuartz:**
   - ติดตั้งและเปิด **XQuartz**
   - ไปที่ **Preferences** > **Security** > ติ๊กถูก `Allow connections from network clients`
   - **Restart** เครื่อง Mac หรือ Log out/Log in 1 รอบ

2. **Run Command:**
   ```bash
   # อนุญาตการเชื่อมต่อ X11
   xhost + 127.0.0.1
   
   # รัน Container
   docker run -it --rm -e DISPLAY=host.docker.internal:0 ai-hand-tracker
   ```
   *(หมายเหตุ: บน macOS, Docker Desktop ยังไม่รองรับการส่งผ่าน Webcam (USB Passthrough) โดยตรงได้สมบูรณ์ หากกล้องไม่ขึ้น แนะนำให้รันแบบ Local Python หรือใช้ Network Camera)*

#### 🐧 Linux (Ubuntu/Debian)
วิธีนี้ง่ายที่สุดและรองรับ Native Hardware
1. **Allow X11 Access:**
   ```bash
   xhost +local:docker
   ```
2. **Run Command:**
   ```bash
   docker run -it --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ai-hand-tracker
   ```

### 🐳 Docker Compose (Recommended / วิธีแนะนำ)
**[TH]** เพื่อความสะดวก ไม่ต้องพิมพ์คำสั่งยาวๆ สามารถใช้ `docker-compose` ได้เลย
ไฟล์ `docker-compose.yml` ได้เตรียมการตั้งค่าไว้แล้ว ให้นักศึกษาเข้าไปแก้ไขไฟล์เพื่อเลือก OS ของตนเองก่อนใช้งาน

1. **แก้ไขไฟล์ `docker-compose.yml` (เฉพาะ Mac/Linux)**
   - **Windows:** ใช้งานได้เลย ไม่ต้องแก้ไข
   - **macOS:** ต้องเข้าไปปิด Comment ของ Windows และเปิด Comment ของ macOS แทน

2. **รันคำสั่ง**
   ```bash
   docker-compose up --build
   ```

---
**© 2024 Nakarin Sripanya.** All Rights Reserved.
