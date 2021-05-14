import cv2

face=cv2.CascadeClassifier('/Users/sayantanpal/Downloads/Homework/haarcascade_frontalface_default.xml')
eye =cv2.CascadeClassifier('/Users/sayantanpal/Downloads/Homework/haarcascade_eye.xml')
smile=cv2.CascadeClassifier('/Users/sayantanpal/Downloads/Homework/haarcascade_smile.xml')

def detect(gray,original):
    faces=face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(original,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey= gray[y:y+h,x:x+w]
        roi_original=original[y:y+h,x:x+w]
        eyes=eye.detectMultiScale(roi_grey,1.1,22) 
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_original,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles=smile.detectMultiScale(roi_grey,1.7,22)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_original,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)           
    return [original,gray]

video_capture=cv2.VideoCapture(0)
while(True):
    _,original = video_capture.read()
    gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,original)
    cv2.imshow('Video',canvas[1])
    cv2.imshow('Video',canvas[0])

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


