import cv2
import time

count = 1

facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    originalimg = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facecascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        coord = [x,y,w,h]

    # Display the resulting frame
    cv2.imshow('Video', frame)

    #reading keys
    key = cv2.waitKey(1)
    if key == ord('s'):
        #first crop image then save
        roi_img = originalimg[coord[1]: coord[1]+coord[3] , coord[0]: coord[0]+coord[2]]
        cv2.imwrite("dataset/saksham/{}.jpg".format(count),roi_img)
        count += 1
    elif key == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()