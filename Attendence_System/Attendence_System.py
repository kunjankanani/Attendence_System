import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# m = face_recognition.load_image_file("photos/kunjan.jpg")
# x = face_recognition.face_encodings(m)[0]

path = 'photos'

images = []
classNames = os.listdir(path)
mylist = os.listdir(path)

for cl in mylist:
    curImg = face_recognition.load_image_file(f'{path}/{cl}')
    images.append(curImg)


def findEncodings(images):
    encodeList = []
    for img in images:
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
known_face_encoding = findEncodings(images)

known_faces_names = [
"ratan tata",
"tesla"]

students = known_faces_names.copy()

face_locations = []
face_encoding = []
face_name = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date +'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_fream = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_fream[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations (rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches [best_match_index]:
                name = known_faces_names [best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print("\n"+ name + "\n")
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendence System",frame)
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()