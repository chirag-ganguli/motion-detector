# Chirag Ganguli

import cv2

static_back = None
video = cv2.VideoCapture(0) 

while True: 
    get, canvas = video.read() 
    motion = 0
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 

    if static_back is None: 
        static_back = gray 
        continue
    diff = cv2.absdiff(static_back, gray) 
    threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1] 
    threshold = cv2.dilate(threshold, None, iterations = 2) 

    cnts,_ = cv2.findContours(threshold.copy(),  
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        motion = 1
        (x, y, z, h) = cv2.boundingRect(contour) 
        cv2.rectangle(canvas, (x, y), (x + z, y + h), (0, 300, 0), 5) 
    cv2.imshow("Motion Detector", canvas) 
    key = cv2.waitKey(1) 
  
video.release() 
cv2.destroyAllWindows() 