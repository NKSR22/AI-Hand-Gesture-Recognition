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

1. **Python (Required):** [Python 3.x for Windows](https://www.python.org/downloads/) (Check "Add Python to PATH")
2. **Git (Required):** [git-scm.com](https://git-scm.com/download/win)

### 🐧 For Linux Users
1. **Python 3 & Pip:** `sudo apt-get update && sudo apt-get install python3 python3-pip`
2. **Git:** `sudo apt-get install git`

---

## 🚀 Installation & Usage (วิธีการติดตั้งและใช้งาน)

### 🔵 Option 1: Run with Virtual Environment (venv - Recommended Example)
**วิธีที่ 1: ใช้งานผ่าน Virtual Environment (แนะนำสูงสุด)**

1. **Clone Repository:**
   ```bash
   git clone git@github.com:NKSR22/AI-Hand-Gesture-Recognition.git
   cd AI-Hand-Gesture-Recognition
   ```

2. **Create & Activate Virtual Environment:**
   * **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python main.py
   ```

5. **Deactivate Environment (เมื่อเลิกใช้งาน):**
   ```bash
   deactivate
   ```

---

### 🟢 Option 2: Run Locally (Quick Start)
**วิธีที่ 2: รันบนเครื่องโดยตรง (แบบรวดเร็ว)**

1. **Clone & Enter Directory:**
   ```bash
   git clone git@github.com:NKSR22/AI-Hand-Gesture-Recognition.git
   cd AI-Hand-Gesture-Recognition
   ```

2. **Install Dependencies (Global):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

---

### 🐳 Option 3: Run with Docker (Linux Only)
**วิธีที่ 3: รันผ่าน Docker (แนะนำเฉพาะระบบปฏิบัติการ Linux)**
> **⚠️ Warning for Windows & macOS Users:**
> It is **NOT recommended** to use Docker for this project on Windows or macOS. Please use **Option 1 or 2**.

**Instructions for Linux:**

1. **Clone Repository:**
   ```bash
   git clone git@github.com:NKSR22/AI-Hand-Gesture-Recognition.git
   cd AI-Hand-Gesture-Recognition
   ```

2. **Setup X11:**
   ```bash
   xhost +local:docker
   ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

#### Manage Docker Containers
- **Stop:** กด `Ctrl + C` หรือ `docker-compose stop`
- **Restart:** `docker-compose up`
- **Clean Up:** `docker-compose down`

---

### 💻 Usage with VS Code (การใช้งานผ่าน VS Code)

#### แบบที่ 1: ใช้ร่วมกับ venv (แนะนำ)
1. เปิดโฟลเดอร์โปรเจกต์ด้วย VS Code
2. เปิด Terminal ใน VS Code (`Ctrl + \``) และสร้าง venv ตามขั้นตอน **Option 1**
3. **เลือก Interpreter:**
   - กด `Ctrl + Shift + P` (Windows) หรือ `Cmd + Shift + P` (macOS)
   - พิมพ์ `Python: Select Interpreter`
   - เลือกตัวที่มีคำว่า `('venv': venv)`
4. เปิดไฟล์ `main.py` แล้วกดปุ่ม ▶️ **Run** (มุมขวาบน)

#### แบบที่ 2: รันโดยตรง (Direct Run)
1. เปิดโฟลเดอร์โปรเจกต์ด้วย VS Code
2. เปิด Terminal ใน VS Code และลง dependencies
3. เปิดไฟล์ `main.py` แล้วกดปุ่ม ▶️ **Run**

---

## 🔍 Troubleshooting (การแก้ปัญหาที่พบบ่อย)

### ❌ 'python' is not recognized
- **Solution:** ตอนลง Python ไม่ได้ติ๊ก **"Add Python to PATH"** ให้ทำการลง Python ใหม่ หรือเพิ่ม PATH ด้วยตนเอง

### ❌ Camera not opening / Error: cv2.error
- **Solution:**
  1. ตรวจสอบว่าไม่มีโปรแกรมอื่นใช้งานกล้องอยู่
  2. ลองถอดเสียบสาย USB กล้องใหม่
  3. ตรวจสอบ Privacy Settings ใน Windows ว่าอนุญาตให้แอปเข้าถึงกล้องหรือไม่

---
**© 2024 Nakarin Sripanya.** All Rights Reserved.
