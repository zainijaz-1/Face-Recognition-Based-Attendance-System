# 📌 Face Recognition-Based Attendance System

This project is an automated attendance management system that uses Face Recognition technology to mark student attendance in real time. The system is built with Python, OpenCV, Streamlit, and SQLite, providing a secure and efficient way to manage student records without the need for manual roll calls or RFID cards.

## ✨ Features

🎥 Live Face Detection & Recognition using Haar Cascade + LBPH algorithm.
🕒 Time-bound Attendance – marks students as Present or Late based on predefined time ranges.
👨‍💻 Admin Panel for managing students, training the recognition model, and setting attendance time slots.
📊 Attendance Reports can be viewed in a dashboard and exported in CSV format.
🔐 Secure Login for administrators with validation stored in SQLite.
💾 Database Integration (SQLite) for storing student data, face datasets, attendance logs, and admin credentials.

## ⚙ Tech Stack

Frontend: Streamlit (Interactive UI)
Backend: Python + OpenCV
Database: SQLite
Libraries: NumPy, Pandas, OpenCV, Streamlit

## 🚀 How It Works

1. Admin registers students and captures their face data.
2. The recognition model is trained and updated automatically.
3. During attendance marking, the system scans faces in real time.
4. Attendance is recorded in the database as Present or Late.
5. Reports can be generated and exported for analysis.

## 🛠 Installation

### Clone the repository:

```
git clone https://github.com/your-username/Face-Recognition-Based-Attendance-System.git
cd Face-Recognition-Based-Attendance-System
```

### Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate          # On Linux/Mac
venv\Scripts\activate             # On Windows
```

### Install dependencies from requirements.txt:

```
pip install -r requirements.txt
```
## ▶ How to Run

Since the project is built with Streamlit, you can launch it with:
```
streamlit run app.py
```
This will open the system in your default web browser where you can:
Login as Admin
Register new students
Start attendance marking
View or download attendance reports

## 📜 License

This project is licensed under the MIT License – you are free to use, modify, and distribute it with proper attribution.
⚡ Now it’s ready for GitHub – clear description, features, and direct run command
