import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

# Path to the folder containing images
path = 'image_basics'
images = []
classnames = []

# Filter only valid image files
valid_extensions = {".jpg", ".jpeg", ".png"}
myList = os.listdir(path)
print(myList)
for cl in myList:
    if os.path.splitext(cl)[1].lower() in valid_extensions:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classnames.append(os.path.splitext(cl)[0])
        else:
            print(f"Warning: Unable to read file {cl}. Skipping...")
    else:
        print(f"Warning: Skipping non-image file {cl}")

# Check if any valid images were found
if not images:
    print("No valid images found in the folder. Please check the 'image_basics' folder.")
    exit()

print(classnames)

# Function to find encodings of the given images
def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if a face was detected
            encodeList.append(encodings[0])
        else:
            print(f"Warning: No face detected in image '{classnames[idx]}'. Skipping...")
    return encodeList

# Function to mark attendance in a CSV file
def markAttendence(name):
    with open('attendence.csv', 'a+') as f:
        f.seek(0)  # Go to the start of the file to read data
        myDataList = f.readlines()  # Read all lines
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Encode known images
encodeListKnown = findEncodings(images)
print('Encoding complete')

# Open the webcam and start face recognition
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrames = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceloc in zip(encodeCurFrames, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        facedis = face_recognition.face_distance(encodeListKnown, encodeFace)

        print(facedis)
        if len(facedis) > 0:
            matchIndex = np.argmin(facedis)
            if matches[matchIndex]:
                name = classnames[matchIndex].upper()
                print(name)

                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                markAttendence(name)
        else:
            print("No matches found")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
