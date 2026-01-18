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

## 🛠 Installation & Usage (วิธีการติดตั้งและใช้งาน)
**Choose one of the following methods:** เลือกใช้วิธีใดวิธีหนึ่ง (แนะนำวิธีที่ 1 สำหรับผู้เริ่มต้น)

### 🟢 Option 1: Run Locally (รันบนเครื่องโดยตรง - Recommended)
**Prerequisites:** Python 3.9+, Webcam

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

---

### 🐳 Option 2: Run with Docker (รันผ่าน Docker - Advanced)
**[EN]** Running GUI applications with Webcam access in Docker requires **X11 Forwarding** configuration.
**[TH]** การรันโปรแกรมที่มี GUI และ Webcam ผ่าน Docker จำเป็นต้องตั้งค่า X11 Forwarding เพื่อให้ Container สามารถแสดงผลหน้าต่างโปรแกรมบนเครื่องของเราได้

#### 0. Install Docker (ติดตั้ง Docker)
ก่อนเริ่มใช้งาน ต้องติดตั้ง Docker Desktop บนเครื่องของท่าน
- **Download:** [Docker Desktop for Windows/Mac/Linux](https://www.docker.com/products/docker-desktop/)
- **Verify:** เปิด Terminal/CMD แล้วพิมพ์คำสั่ง `docker --version` หากขึ้นเลขเวอร์ชันแสดงว่าติดตั้งสำเร็จ

#### 1. Clone Repository (ดาวน์โหลดไฟล์โปรเจกต์)
ก่อนจะรัน Docker Compose คุณต้องมีไฟล์โปรเจกต์ในเครื่องก่อน

```bash
git clone https://github.com/NKSR22/AI-Hand-Gesture-Recognition.git
cd AI-Hand-Gesture-Recognition
```

#### 2. Setup X11/Display (การเตรียมเครื่อง)
*ทำการตั้งค่าตาม OS ของท่านก่อน*

**🪟 Windows (WSL2)**
- **Requirement:** [VcXsrv](https://sourceforge.net/projects/vcxsrv/) & [usbipd-win](https://github.com/dorssel/usbipd-win)
- **VcXsrv:** เปิดโปรแกรมเลือก `Multiple windows` > `Start no client` > ติ๊กถูก `Disable access control`
- **WSL:** เชื่อมต่อกล้องผ่าน `usbipd wsl attach ...`

**🍎 macOS**
- **Requirement:** [XQuartz](https://www.xquartz.org/)
- **Setup:** เปิด XQuartz > Preferences > Security > ติ๊กถูก `Allow connections from network clients` > Restart Mac
- **Command:** รันคำสั่ง `xhost + 127.0.0.1`

**🐧 Linux**
- **Command:** รันคำสั่ง `xhost +local:docker`

#### 3. Run with Docker Compose (รันโปรแกรม)
**[TH]** ไฟล์ `docker-compose.yml` ได้เตรียมการตั้งค่าไว้แล้ว (Default สำหรับ Windows)

1. **แก้ไขไฟล์ `docker-compose.yml` (เฉพาะ Mac/Linux)**
   - **Windows:** ใช้งานได้เลย ไม่ต้องแก้ไข
   - **macOS:** ต้องเข้าไปปิด Comment ของ Windows และเปิด Comment ของ macOS แทน

2. **รันคำสั่ง**
   ```bash
   docker-compose up --build
   ```

#### 4. Develop & Edit Code (การพัฒนาและแก้ไขโค้ด)
เนื่องจากเราได้ทำการเชื่อมต่อไฟล์ (Volume Mount) ระหว่างเครื่องจริงกับ Docker ไว้แล้ว (`.:/app`)

**วิธีที่ 1: แก้ไขผ่าน VS Code (แนะนำ)**
1. เปิดไฟล์ `main.py` บนเครื่องคอมพิวเตอร์ของคุณ (Windows/Mac) ด้วย VS Code หรือ Text Editor ที่ถนัด
2. แก้ไขโค้ดและกดบันทึก (Save)
3. รันคำสั่ง `docker-compose restart` เพื่อเริ่มโปรแกรมใหม่ด้วยโค้ดล่าสุด

**วิธีที่ 2: แก้ไขใน Terminal ของ Docker**
หากต้องการเข้าไปแก้ไขไฟล์ข้างใน Container โดยตรง:
1. เช็ค ID ของ Container: `docker ps`
2. เข้าไปใน Container: `docker exec -it ai-hand-gesture-container bash`
3. ติดตั้ง Nano (ครั้งแรก): `apt-get update && apt-get install -y nano`
4. แก้ไขไฟล์: `nano main.py` (กด `Ctrl+X` > `Y` > `Enter` เพื่อบันทึก)
5. รันโปรแกรมใหม่: `python main.py`

---
**© 2024 Nakarin Sripanya.** All Rights Reserved.
