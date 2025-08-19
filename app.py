import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime, time
import pandas as pd
import sqlite3

# Directories & Files
DATASET_DIR = 'dataset'
TRAINER_DIR = 'trainer'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DB_NAME = 'attendance.db'
ADMIN_DB = 'admin.db'
ATTENDANCE_FILE = 'attendance.csv'

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Initialize Databases
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date TEXT, time TEXT, status TEXT)''')
    conn.commit()
    conn.close()

def init_admin_db():
    conn = sqlite3.connect(ADMIN_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS admin (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()
init_admin_db()

# Attendance Marking Function
def mark_attendance(name, status):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time_now = now.strftime('%H:%M:%S')

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])

    if not ((df['Name'] == name) & (df['Date'] == date)).any():
        new_row = {'Name': name, 'Date': date, 'Time': time_now, 'Status': status}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)', (name, date, time_now, status))
        conn.commit()
        conn.close()

def admin_signup():
    st.markdown("""<style>.stTextInput>div>div>input{background:#0a192f;color:white;border:1px solid #1e88e5;}
    .stButton>button{background:#1e88e5!important;color:white!important;}</style>""", unsafe_allow_html=True)
    
    st.subheader("Admin Sign Up")
    username, password = st.text_input("Username"), st.text_input("Password", type="password")
    
    if st.button("Sign Up"):
        with sqlite3.connect(ADMIN_DB) as conn:
            if conn.execute('SELECT 1 FROM admin WHERE username=?', (username,)).fetchone():
                st.error("Admin exists!")
            else:
                conn.execute('INSERT INTO admin VALUES (?,?)', (username, password))
                st.success("Registered!")

# Admin Login
def admin_login():
    st.subheader("Admin Login")
    username = st.text_input("Username", key='login_user')
    password = st.text_input("Password", type="password", key='login_pass')
    if st.button("Login"):
        conn = sqlite3.connect(ADMIN_DB)
        c = conn.cursor()
        c.execute('SELECT * FROM admin WHERE username = ? AND password = ?', (username, password))
        if c.fetchone():
            st.session_state['admin_logged_in'] = True
            st.success("Logged in as Admin.")
            st.rerun()
        else:
            st.error("Invalid Credentials")
        conn.close()

# Admin Dashboard
def admin_dashboard():
    st.sidebar.title("Admin Dashboard")
    admin_page = st.sidebar.radio("Admin Panel", ['Add Student', 'Delete Student', 'Train Model', 'Set Attendance Time', 'View Attendance'])

    if admin_page == 'Add Student':
        add_student()
    elif admin_page == 'Delete Student':
        delete_student()
    elif admin_page == 'Train Model':
        train_model()
    elif admin_page == 'Set Attendance Time':
        set_attendance_time()
    elif admin_page == 'View Attendance':
        view_attendance()

# Add Student
def add_student():
    st.header("Register New Student")
    username = st.text_input("Enter Student Name")
    start_button = st.button("Capture Faces")
    if username and start_button:
        user_dir = os.path.join(DATASET_DIR, username)
        os.makedirs(user_dir, exist_ok=True)
        cap = cv2.VideoCapture(0)
        img_count = 1
        st.info("Press 's' to save face | Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Capture Faces - Press 's' to save, 'q' to quit", frame)
            key = cv2.waitKey(1)
            if key == ord('s') and len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                face_path = os.path.join(user_dir, f"{username}_{img_count}.jpg")
                cv2.imwrite(face_path, face_img)
                img_count += 1
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        st.success("Face samples saved.")

# Delete Student
def delete_student():
    st.header("Delete Student Record")
    students = os.listdir(DATASET_DIR)
    if students:
        student_to_delete = st.selectbox("Select Student", students)
        if st.button("Delete Student"):
            student_dir = os.path.join(DATASET_DIR, student_to_delete)
            for file in os.listdir(student_dir):
                os.remove(os.path.join(student_dir, file))
            os.rmdir(student_dir)
            st.success(f"Deleted {student_to_delete}'s data.")

            # Retrain Model after Deletion
            st.info("Retraining Model after deletion...")
            train_model(retrain=True)
    else:
        st.info("No students found.")

# Train Model
def train_model(retrain=False):
    if not retrain:
        st.header("Train Face Recognition Model")
    if retrain or st.button("Start Training"):
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
        label_map = {}
        face_samples, ids = [], []
        current_id = 1
        for folder in os.listdir(DATASET_DIR):
            folder_path = os.path.join(DATASET_DIR, folder)
            if os.path.isdir(folder_path):
                label_map[current_id] = folder
                for image_file in os.listdir(folder_path):
                    try:
                        img = Image.open(os.path.join(folder_path, image_file)).convert('L')
                        img_np = np.array(img, 'uint8')
                        faces = face_cascade.detectMultiScale(img_np)
                        for (x, y, w, h) in faces:
                            face_samples.append(img_np[y:y+h, x:x+w])
                            ids.append(current_id)
                    except:
                        continue
                current_id += 1

        if len(face_samples) == 0:
            st.warning("No faces found to train. Please add student data.")
            return

        recognizer.train(face_samples, np.array(ids))
        recognizer.write(os.path.join(TRAINER_DIR, 'trainer.yml'))
        with open(os.path.join(TRAINER_DIR, 'labels.txt'), 'w') as f:
            for id, name in label_map.items():
                f.write(f"{id}:{name}\n")
        st.success("Model trained successfully.")

# Set Attendance Time using session_state
if 'attendance_start_time' not in st.session_state:
    st.session_state.attendance_start_time = time(9, 0)
if 'attendance_end_time' not in st.session_state:
    st.session_state.attendance_end_time = time(9, 15)

def set_attendance_time():
    st.header("Set Attendance Time")
    start_time = st.time_input("Start Time", value=st.session_state.attendance_start_time)
    end_time = st.time_input("End Time", value=st.session_state.attendance_end_time)
    if st.button("Set Time"):
        st.session_state.attendance_start_time = start_time
        st.session_state.attendance_end_time = end_time
        st.success("Time Successfully Updated!")

# View Attendance
def view_attendance():
    st.header("Attendance Report")
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT name AS Name, date AS Date, time AS Time, status AS Status FROM attendance", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="attendance_report.csv", mime='text/csv')
    else:
        st.warning("No attendance records found.")

def mark_attendance_page():
    # Minimal styling
    st.markdown("""
    <style>
    .stApp {background-color: #0a192f; color: white;}
    .stButton>button {background: #1e88e5; color: white; border-radius: 5px;}
    h1 {color: #1e88e5 !important;}
    </style>
    """, unsafe_allow_html=True)

    st.header("Live Attendance Marking")
    
    if st.button("Start Attendance"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(os.path.join(TRAINER_DIR, 'trainer.yml'))
        
        # Load labels
        labels = {int(id_str): name for line in open(os.path.join(TRAINER_DIR, 'labels.txt')) 
                 for id_str, name in [line.strip().split(':')]}
        
        cap = cv2.VideoCapture(0)
        st.warning("Press ESC to stop")
        
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.2, 5):
                id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 80:
                    name = labels.get(id_, "Unknown")
                    status = "Present" if st.session_state.attendance_start_time <= datetime.now().time() <= st.session_state.attendance_end_time else "Late"

                    mark_attendance(name, status)
                    color = (0, 255, 0) if status == "Present" else (0, 165, 255)
                else:
                    color = (0, 0, 255)
                    
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({status})" if conf < 80 else "Unknown", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        st.success("Done!")
        
# ------------------- Main -------------------- #
st.set_page_config(page_title="Face Attendance Management System", layout="wide")

if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False

page = st.sidebar.selectbox("Select Page", ["Attendance System", "Admin Authentication"])

if page == "Attendance System":
    st.title("Online Attendance")
    mark_attendance_page()

elif page == "Admin Authentication":
    if st.session_state['admin_logged_in']:
        st.sidebar.success("Logged in as Admin")
        if st.sidebar.button("Logout"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
        admin_dashboard()
    else:
        tabs = st.tabs(["Login", "Sign Up"])
        with tabs[0]:
            admin_login()
        with tabs[1]:
            admin_signup()
