# AI Hand Tracking & Gesture Recognition (Teachable Machine)

**Real-time Hand Gesture Recognition System using MediaPipe & KNN Classifier**
**ระบบจำแนกท่าทางมือแบบเรียลไทม์ด้วย MediaPipe และ KNN Classifier**

---

### 🌐 Project Information (ข้อมูลโปรเจกต์)

**Subject:** Introduction to Artificial Intelligence (รายวิชาปัญญาประดิษฐ์เบื้องต้น)  
**Program:** B.Ind.Tech in Electrical Technology and Intelligence Control Systems (หลักสูตร อส.บ.เทคโนโลยีไฟฟ้าและระบบควบคุมอัจฉริยะ)  
**Department:** Electrical Engineering, Faculty of Industry and Technology  
**University:** Rajamangala University of Technology Isan, Sakon Nakhon Campus (มหาวิทยาลัยเทคโนโลยีราชมงคลอีสาน วิทยาเขตสกลนคร)

**Developer / Instructor:** Nakarin Sripanya (อ.นครินทร์ ศรีปัญญา)  
**Email:** <nakatin.sr@rmuti.ac.th>  
**GitHub:** [https://github.com/NKSR22

---

## 📖 About the Project (เกี่ยวกับโปรเจกต์)

**[EN]** This project is a real-time hand gesture recognition system developed using Python, MediaPipe, and OpenCV. It features a **"Teachable Machine"** capability, allowing users to train the AI to recognize custom hand gestures instantly using a K-Nearest Neighbors (KNN) algorithm without retraining the entire model.

**[TH]** โปรเจกต์นี้เป็นระบบตรวจจับมือและนับนิ้วแบบเรียลไทม์ พัฒนาด้วย Python, MediaPipe และ OpenCV จุดเด่นคือความสามารถแบบ **"Teachable Machine"** ที่เปิดโอกาสให้ผู้ใช้สามารถ "สอน" AI ให้จดจำท่าทางมือใหม่ๆ ได้ทันทีผ่านอัลกอริทึม KNN โดยไม่ต้องเทรนโมเดลใหม่ตั้งแต่ต้น

---

## 🧠 Theory & Principles (ทฤษฎีและหลักการทำงาน)

### 1. Computer Vision & Landmark Detection (การตรวจจับจุดพิกัดมือ)

**[EN]** The system utilizes **MediaPipe Hands**, a high-fidelity hand tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from a single video frame.
- **Palm Detection Model:** Operates on the full image and returns an oriented hand bounding box.
- **Hand Landmark Model:** Operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints.

**[TH]** ระบบใช้ **MediaPipe Hands** ซึ่งเป็นโซลูชันการติดตามมือที่มีความแม่นยำสูง โดยใช้ Machine Learning ในการคาดการณ์จุดพิกัด 3 มิติ (3D Landmarks) จำนวน 21 จุดบนมือจากภาพวิดีโอ
- **โมเดลตรวจจับฝ่ามือ:** ค้นหาตำแหน่งมือในภาพรวม
- **โมเดลหาจุดพิกัดมือ:** ทำงานในกรอบภาพที่ตัดเฉพาะส่วนมือ เพื่อหาตำแหน่งข้อต่อต่างๆ อย่างละเอียด

![Hand Landmarks](https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png)

### 2. Feature Extraction (การสกัดคุณลักษณะ)

**[EN]** Raw coordinates (x, y) from the camera are not suitable for direct classification due to position and scale variations. The system performs preprocessing:
- **Translation Invariance:** All points are shifted relative to the Wrist (Point 0).
- **Scale Invariance:** Coordinates are normalized by the hand's size to ensure consistent recognition regardless of distance.

**[TH]** พิกัดดิบ (x, y) จากกล้องไม่เหมาะแก่การนำไปจำแนกโดยตรงเนื่องจากความแปรปรวนของตำแหน่งและขนาด ระบบจึงทำการประมวลผลเบื้องต้น:
- **ไม่ขึ้นกับตำแหน่ง (Translation Invariance):** ย้ายจุดอ้างอิงทั้งหมดไปที่ข้อมือ (จุดที่ 0) เพื่อให้การขยับมือไปมาบนหน้าจอไม่มีผลต่อการจำแนก
- **ไม่ขึ้นกับขนาด (Scale Invariance):** ปรับขนาดพิกัดเทียบกับขนาดฝ่ามือ เพื่อให้สามารถจำแนกท่าทางได้ไม่ว่ามือจะอยู่ใกล้หรือไกลกล้อง

### 3. K-Nearest Neighbors (KNN) Classification (การจำแนกด้วย KNN)

**[EN]** For the "Teachable" feature, the system uses the **K-Nearest Neighbors (KNN)** algorithm.
- When a user saves a gesture, the extracted feature vector is stored.
- During inference, the system calculates the **Euclidean Distance** between the current hand pose and all stored samples.
- It selects the `K` closest samples and assigns the label based on a majority vote.

**[TH]** สำหรับฟีเจอร์ "สอนได้" ระบบใช้อัลกอริทึม **K-Nearest Neighbors (KNN)**
- เมื่อผู้ใช้บันทึกท่าทาง ข้อมูลเวกเตอร์คุณลักษณะ (Feature Vector) จะถูกเก็บในหน่วยความจำ
- ในขั้นตอนการทำนาย ระบบจะคำนวณ **ระยะห่างยูคลิด (Euclidean Distance)** ระหว่างท่ามือปัจจุบันกับข้อมูลตัวอย่างทั้งหมดที่เก็บไว้
- ระบบจะเลือกตัวอย่างที่ใกล้เคียงที่สุดจำนวน `K` ตัว และตัดสินใจเลือกคำตอบจากเสียงข้างมาก (Majority Vote)

---

## 🛠 Prerequisites (เตรียมความพร้อมก่อนเริ่ม)

### 🪟 For Windows Users (สำหรับผู้ใช้ Windows)
**[EN]** If you are using a new Windows installation, install tools in this order:
1. **Python (Required):** Download from [python.org](https://www.python.org/downloads/). **Check "Add Python to PATH"** during installation.
2. **Git (Required):** Download from [git-scm.com](https://git-scm.com/).

**[TH]** สำหรับผู้ที่เพิ่งลง Windows ใหม่ หรือยังไม่มีเครื่องมือ ให้ทำการติดตั้งตามลำดับดังนี้:
1. **Python (จำเป็น):** ดาวน์โหลดจาก [python.org](https://www.python.org/downloads/) และ **ต้องติ๊ก "Add Python to PATH"** ขณะติดตั้ง
2. **Git (จำเป็น):** ดาวน์โหลดจาก [git-scm.com](https://git-scm.com/)

### 🐧 For Linux Users (สำหรับผู้ใช้ Linux)
**[EN]** Update your system and install necessary packages:
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip git
```

**[TH]** อัปเดตระบบและติดตั้งแพ็กเกจที่จำเป็น:
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip git
```

---

## 🚀 Installation & Usage (วิธีการติดตั้งและใช้งาน)

### 🔵 Option 1: Virtual Environment (Recommended) / วิธีที่ 1: ทางเลือกที่แนะนำ

**[EN]** Using a virtual environment prevents library conflicts between projects.
1. **Clone Repository:**
   ```bash
   git clone git@github.com:NKSR22/AI-Hand-Gesture-Recognition.git
   cd AI-Hand-Gesture-Recognition
   ```
2. **Create & Activate Virtual Environment:**
   - **Windows:** `python -m venv venv` then `.\venv\Scripts\activate`
   - **Mac/Linux:** `python3 -m venv venv` then `source venv/bin/activate`
3. **Install Dependencies:** `pip install -r requirements.txt`
4. **Run Application:** `python main.py`
5. **Deactivate:** Type `deactivate` when finished.

**[TH]** การใช้สภาพแวดล้อมเสมือนช่วยป้องกันปัญหาไลบรารีตีกันระหว่างโปรเจกต์
1. **ดาวน์โหลดโปรเจกต์:** (ใช้คำสั่ง `git clone` ด้านบน)
2. **สร้างและเปิดใช้งาน venv:**
   - **Windows:** `python -m venv venv` ตามด้วย `.\venv\Scripts\activate`
   - **Mac/Linux:** `python3 -m venv venv` ตามด้วย `source venv/bin/activate`
3. **ติดตั้งไลบรารี:** `pip install -r requirements.txt`
4. **เริ่มโปรแกรม:** `python main.py`
5. **การปิด:** พิมพ์ `deactivate` เมื่อเลิกใช้งาน

---

### 🟢 Option 2: Run Locally (Quick Start) / วิธีที่ 2: รันบนเครื่องโดยตรง

**[EN]** Best for quick testing without virtual environments.
1. **Clone & Enter Folder:** (See Option 1)
2. **Install Globally:** `pip install -r requirements.txt`
3. **Run Application:** `python main.py`

**[TH]** เหมาะสำหรับการทดสอบแบบรวดเร็วโดยไม่แยกสภาพแวดล้อม
1. **ดาวน์โหลดโปรเจกต์:** (ดูวิธีในข้อ 1)
2. **ติดตั้งไลบรารี:** `pip install -r requirements.txt`
3. **เริ่มโปรแกรม:** `python main.py`

---

### 🐳 Option 3: Run with Docker (Linux ONLY) / วิธีที่ 3: รันผ่าน Docker (เฉพาะ Linux)

**[EN]** **Warning:** Not recommended for Windows/Mac due to camera passthrough complexity.
1. **Clone Project:** (See Option 1)
2. **Setup X11 Access:** `xhost +local:docker`
3. **Run:** `docker-compose up --build`

**[TH]** **คำเตือน:** ไม่แนะนำสำหรับ Windows/Mac เนื่องจากความยุ่งยากในการเชื่อมต่อกล้อง
1. **ดาวน์โหลดโปรเจกต์:** (ดูวิธีในข้อ 1)
2. **ตั้งค่าหน้าจอ:** `xhost +local:docker`
3. **รันโปรแกรม:** `docker-compose up --build`

---

## 💻 Usage with VS Code (การใช้งานผ่าน VS Code)

**[EN]** Visual Studio Code is the recommended editor for this project.
- **Using with venv:** Open project > Open Terminal > Create venv > Press `Ctrl+Shift+P` > `Python: Select Interpreter` > Choose `venv`.
- **Direct Run:** Install dependencies via terminal > Open `main.py` > Click ▶️ **Run**.

**[TH]** แนะนำใช้ Visual Studio Code ในการพัฒนา
- **ใช้งานร่วมกับ venv:** เปิดโฟลเดอร์ > เปิด Terminal > สร้าง venv > กด `Ctrl+Shift+P` > เลือก `Python: Select Interpreter` > เลือก `venv`
- **รันโดยตรง:** ติดตั้งไลบรารีผ่าน Terminal > เปิดไฟล์ `main.py` > กดปุ่ม ▶️ **Run**

---

## 🔍 Troubleshooting (การแก้ปัญหาที่พบบ่อย)

### ❌ 'python' is not recognized
- **[EN]** Python was not added to your system PATH. Reinstall Python and check the "Add to PATH" box.
- **[TH]** ระบบหาคำสั่ง python ไม่เจอ เนื่องไม่ได้ติ๊ก "Add Python to PATH" ขณะติดตั้ง ให้ทำการแก้ไข PATH หรือติดตั้งใหม่

### ❌ Camera not opening / Error: cv2.error
- **[EN]** Check if another app is using the camera (Zoom, Teams). Try replugging your USB camera or check Windows Privacy settings.
- **[TH]** กล้องใช้งานไม่ได้ ตรวจสอบว่าไม่มีโปรแกรมอื่นแอบใช้กล้องอยู่ (เช่น Zoom) หรือลองตรวจสอบการอนุญาตเข้าถึงกล้องในหน้า Privacy ของ Windows

---
**© 2024 Nakarin Sripanya.** All Rights Reserved.
