
Attendance Management System
This project automates attendance tracking and ensures no proxy entries using two main scripts:

attendence.py: Tracks attendance based on facial recognition.
counting.py: Verifies the number of individuals to detect potential proxy attendance.
Features
1. attendence.py
Tracks attendance using a webcam or video feed.
Detects and recognizes faces in real-time.
Marks attendance with a timestamp and stores the data in a file or database.
Ensures efficient and accurate attendance logging for employees, students, or participants.
2. counting.py
Monitors the number of people entering the area.
Compares the actual count of individuals with the number of detected faces in the attendance log.
Flags any discrepancies to prevent proxy attendance.
Prerequisites
Python 3.7+
Required Python libraries:
OpenCV (cv2)
NumPy
Pandas
face_recognition (if used for facial recognition)
A webcam or external camera for real-time detection.
How to Use
Step 1: Install Dependencies
Install the required Python libraries using pip:

pip install opencv-python numpy pandas face_recognition
Step 2: Run the Scripts
Track Attendance: Run attendence.py to track attendance using facial recognition.

python attendence.py
The script will recognize registered faces and mark attendance.
Attendance logs will be stored as a CSV or in a database.
Verify Proxy Attendance: Run counting.py to monitor the number of individuals.

python counting.py
Compares the count of attendees with the attendance log.
Flags potential proxies for further review.
Output
attendence.py
Generates an attendance log with:
Name
Date
Time
Stored in a CSV file (e.g., attendance.csv) or a database.
counting.py
Displays the actual count of individuals.
Reports any discrepancies with attendance data.
Notes
Ensure that facial data is pre-registered for accurate recognition.
Place the camera at an appropriate angle for optimal detection.
Update counting.py logic to match the entry/exit setup.
Applications
Schools and Universities: Automate student attendance and prevent proxies.
Offices: Track employee attendance with anti-proxy measures.
Events: Monitor participants efficiently.










