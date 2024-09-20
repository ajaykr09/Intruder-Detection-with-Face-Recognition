
from urllib.request import urlopen

from ssl import SSLContext,PROTOCOL_TLSv1

import numpy as np

# Import OpenCV2 for image processing
import cv2

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id
face_id = 1

# Initialize sample face image+
count = 0

# Ip of the IP webcam server (on phone). The phone and your computer must be in the same LAN (connected to the same WiFi)
url = 'https://192.168.1.70:8080/shot.jpg'

# Start looping
while(True):

    # Read the video frame from the url
    gcontext = SSLContext(PROTOCOL_TLSv1)  # Only for gangstars
    info = urlopen(url, context=gcontext).read()

    imgNp=np.array(bytearray(info),dtype=np.uint8)
    image_frame=cv2.imdecode(imgNp,-1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        print(count)
        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break

    # If image taken reach 100, stop taking video
    elif count>100:
        break
