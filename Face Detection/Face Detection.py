#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
vc = cv.VideoCapture(0)
while True:
    _,frame = vc.read()
    gray =cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 4)
    for x, y, w, h in faces:
        cv.rectangle(frame,(x,y), (x+w,y+h),(255, 255, 0), 3)
    cv.imshow('Windos',frame)
    key =cv.waitKey(1)
    if key == ord('x'):
        break
cv.destroyAllWindows()
vc.release()


# In[ ]:




