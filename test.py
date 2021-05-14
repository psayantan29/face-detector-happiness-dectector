import cv2

face = cv2.CascadeClassifier('/Users/sayantanpal/Downloads/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')
eye=cv2.CascadeClassifier('/Users/sayantanpal/Downloads/Module_1_Face_Recognition/haarcascade_eye.xml')

def detect(gray,original):
    faces= face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(original,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = original[y:y+h,x:x+w]
        eyes= eye.detectMultiScale(roi_gray,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
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

