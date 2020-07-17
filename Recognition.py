2020

@author: Abhijeet
&quot;&quot;&quot;

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(&#39;haarcascade_frontalface_default.xml&#39;)
classifier =load_model(&#39;face_62_500.h5&#39;)

class_labels = [&#39;abhijeet&#39;,&#39;Adish Sthalekar&#39;,&#39;aditya chaudhary&#39;,&#39;aditya
mukund&#39;,&#39;Akash&#39;,&#39;anushka&#39;,&#39;arnold&#39;,&#39;ashish&#39;,&#39;bhoomi&#39;,
&#39;Bhushan
Kolhe&#39;,&#39;brayan&#39;,&#39;chaitali&#39;,&#39;dane&#39;,&#39;Divya&#39;,&#39;divyashree&#39;,&#39;dube&#39;,&#39;elton&#39;,&#39;goofran&#39;,&#39;harsh&#39;,&#39;Harshal Rathod&#39;,
&#39;abhi&#39;,&#39;harshala&#39;,&#39;kevin&#39;,&#39;madhura&#39;,&#39;manisha&#39;,&#39;maria&#39;,&#39;mihir&#39;,&#39;mrunal&#39;,&#39;NeilJason&#39;,
&#39;nikita&#39;,&#39;nilesh&#39;,&#39;osama&#39;,&#39;pooja&#39;,&#39;Poonam Mam&#39;,&#39;pransu&#39;,&#39;prathmesh&#39;,&#39;pratik
shetty&#39;,&#39;RadaliaDSouza&#39;,
&#39;Rohit&#39;,&#39;ruban&#39;,&#39;shrey&#39;,&#39;Rushikesh&#39;,&#39;ryan&#39;,&#39;salman&#39;,&#39;Samuel Pais&#39;,&#39;Satish Sir&#39;,&#39;sayali&#39;,&#39;sharayu&#39;,
&#39;uttu&#39;,&#39;utkarsh&#39;,&#39;Shreyas Kulkarni&#39;,
&#39;shubham&#39;,&#39;siddhant&#39;,&#39;siddharth
pagare&#39;,&#39;sumantu&#39;,&#39;umesh&#39;,&#39;utkarsh&#39;,&#39;uttam&#39;,&#39;vaishali&#39;,&#39;vaishnavi&#39;,&#39;vedant&#39;,&#39;vivek&#39;]

# def face_detector(img):
# # Convert image to grayscale
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# faces = face_classifier.detectMultiScale(gray,1.3,5)
# if faces is ():
# return (0,0,0,0),np.zeros((48,48),np.uint8),img

# for (x,y,w,h) in faces:
# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# roi_gray = gray[y:y+h,x:x+w]

# try:
# roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
# except:
# return (x,w,y,h),np.zeros((48,48),np.uint8),img
# return (x,w,y,h),roi_gray,img

cap = cv2.VideoCapture(&#39;abhi.mp4&#39;)

while True:
# Grab a single frame of video
ret, frame = cap.read()
labels = []
#gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(frame,1.3,5)

for (x,y,w,h) in faces:
cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
roi_gray = frame[y:y+h,x:x+w]
roi_gray = cv2.resize(roi_gray,(64,64),interpolation=cv2.INTER_AREA)
# rect,face,image = face_detector(frame)

if np.sum([roi_gray])!=0:
roi = roi_gray.astype(&#39;float&#39;)/255.0
roi = img_to_array(roi)
roi = np.expand_dims(roi,axis=0)

# make a prediction on the ROI, then lookup the class

preds = classifier.predict(roi)[0]
label=class_labels[preds.argmax()]
label_position = (x,y)
cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
else:
cv2.putText(frame,&#39;No Face Found&#39;,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
cv2.imshow(&#39;frame&#39;,frame)
if cv2.waitKey(1) &amp; 0xFF == ord(&#39;q&#39;):
break

cap.release()

cv2.destroyAllWindows()