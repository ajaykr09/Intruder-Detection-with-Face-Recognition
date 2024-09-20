import os
import requests
from urllib.request import urlopen

import json
from flask import Flask, Response, render_template
import cv2
import numpy as np
from ssl import SSLContext, PROTOCOL_TLSv1
import firebase_admin
from firebase_admin import credentials, messaging

# Initialize Firebase Admin with service account for other Firebase services
cred = credentials.Certificate("D:\mobile_proj2\mobcomp-cc6db-firebase-adminsdk-fkcu5-7d66e683f0.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Load the trained model

# Load prebuilt model for Frontal Face
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# IP of the IP webcam server (on phone)
url = 'https://192.168.1.67:8080/shot.jpg'




def gen_frames():
    """Generate frames from the webcam and detect faces."""
    while True:
        gcontext = SSLContext(PROTOCOL_TLSv1)
        info = urlopen(url, context=gcontext).read()
        imgNp = np.array(bytearray(info), dtype=np.uint8)
        im = cv2.imdecode(imgNp, -1)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 50:
                Id = "Unknown"
            else:
                if Id == 2:
                    Id = "Prateek"
                elif Id == 1:
                    Id = "Ajay"
                # Send a notification if a recognized face is detected
                send_fcm_message("device_token_here", "Face Detected", f"{Id} was detected.")

            cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
            cv2.putText(im, str(Id), (x, y - 40), font, 2, (255, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
