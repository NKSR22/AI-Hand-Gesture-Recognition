# AI Hand Tracking & Gesture Recognition (Teachable Machine)

**Real-time Hand Gesture Recognition System using MediaPipe & KNN Classifier**

---

### 🌐 Project Information

**Subject:** Introduction to Artificial Intelligence (รายวิชาปัญญาประดิษฐ์เบื้องต้น)  
**Program:** B.Ind.Tech in Electrical Technology and Intelligence Control Systems (หลักสูตร อส.บ.เทคโนโลยีไฟฟ้าและระบบควบคุมอัจฉริยะ)  
**Department:** Electrical Engineering, Faculty of Industry and Technology  
**University:** Rajamangala University of Technology Isan, Sakon Nakhon Campus (มหาวิทยาลัยเทคโนโลยีราชมงคลอีสาน วิทยาเขตสกลนคร)

**Developer / Instructor:** Nakarin Sripanya (อ.นครินทร์ ศรีปัญญา)  
**Email:** <nakatin.sr@rmuti.ac.th>  
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

## 🛠 Prerequisites (เตรียมความพร้อมก่อนเริ่ม)

### 🪟 For Windows Users (Fresh Install)

**[TH]** สำหรับผู้ที่เพิ่งลง Windows ใหม่ หรือยังไม่มีเครื่องมือ ให้ทำการติดตั้งตามลำดับดังนี้:

#### 1. Python (Required)

- **Download:** [Python 3.x for Windows](https://www.python.org/downloads/)
- **Installation:** Run installer > **Check "Add Python to PATH"** > Install Now.

#### 2. Git (Required)

- **Download:** [git-scm.com](https://git-scm.com/download/win)

### 🐧 For Ubuntu / Linux Users

**[TH]** สำหรับระบบปฏิบัติการ Ubuntu/Linux ที่ต้องการรันผ่าน Docker:

- **Docker & Docker Compose:**

  ```bash
  sudo apt-get update
  sudo apt-get install docker.io docker-compose-plugin
  ```

- **X11 Server:** (Usually installed by default on Ubuntu Desktop)

---

## 💻 Recommended Editor: VS Code

**[TH]** แนะนำให้ใช้ **Visual Studio Code** ในการพัฒนาและแก้ไขโปรเจกต์นี้

**Recommended Extensions:**

1. **Python (Microsoft):** สำหรับการรันและ Debug ภาษา Python
2. **Pylance:** ช่วยตรวจสอบความถูกต้องของโค้ด
3. **Docker (Microsoft):** (Optional) สำหรับจัดการ Container ง่ายๆ

---

## 🚀 Installation & Usage (วิธีการติดตั้งและใช้งาน)

### 🟢 Option 1: Run Locally (Windows / Mac / Linux)

**Recommended for Windows users** (วิธีที่แนะนำสำหรับ Windows)

1. **Clone Repository:**

   ```bash
   git clone git@github.com:NKSR22/AI-Hand-Gesture-Recognition.git
   cd AI-Hand-Gesture-Recognition
   ```

2. **Install Dependencies:**

   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Run Application:**

   ```bash
   python main.py
   ```

---

### 🐳 Option 2: Run with Docker (Ubuntu / Linux Only)

**[TH]** สำหรับผู้ใช้งาน Ubuntu ที่ต้องการ Environment ที่สะอาดและจัดการง่าย

**1. Setup Display Access**
อนุญาตให้ Docker เข้าถึงหน้าจอ (GUI):

```bash
xhost +local:docker
```

**2. Run with Docker Compose**
รันโปรแกรม:

```bash
docker compose up --build
```

*(หากต้องการหยุด ให้กด Ctrl+C)*

**3. Edit Code**
ไฟล์ในเครื่องจะเชื่อม (Sync) กับใน Container อัตโนมัติ สามารถแก้ไขไฟล์ `main.py` ผ่าน VS Code แล้วรันใหม่ได้เลย

---

## 🔍 Troubleshooting (การแก้ปัญหาที่พบบ่อย)

### ❌ 'python' is not recognized

**[TH]** หากพิมพ์คำสั่ง python แล้วเครื่องไม่รู้จัก

- **Solution:** เกิดจากตอนลง Python ไม่ได้ติ๊ก **"Add Python to PATH"** ให้ทำการลง Python ใหม่ หรือ [เพิ่ม PATH ด้วยตนเอง](https://www.google.com/search?q=add+python+to+path+windows)

### ❌ Camera not opening / Error: cv2.error

**[TH]** กล้องไม่ทำงาน หรือเปิดไม่ได้

- **Solution:**
  1. ตรวจสอบว่าไม่มีโปรแกรมอื่นใช้งานกล้องอยู่ (เช่น Zoom, Teams)
  2. ลองถอดเสียบสาย USB กล้องใหม่
  3. ตรวจสอบ Privacy Settings ใน Windows ว่าอนุญาตให้แอปเข้าถึงกล้องหรือไม่ (Camera Privacy Settings > Allow desktop apps to access your camera)

---
**© 2024 Nakarin Sripanya.** All Rights Reserved.
