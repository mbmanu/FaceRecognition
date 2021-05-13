import cv2 as cv

actors = ['Paul', 'Robert']
haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained.yml')

img = cv.imread(r'/media/manu/DATA/Projects/opencv/assets/test/paul1.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

for (i,j,a,b) in faces:
    faces_roi = gray[j:j+b, i:i+a]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'{actors[label]} with {confidence}% surity')
    cv.putText(img, str(actors[label]), (img.shape[1]//2, img.shape[0]), fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1.0, color = (255,0,0), thickness = 2)
    cv.rectangle(img, (i,j), (i+a,j+b), (255,0,0), thickness = 2)
    cv.imshow('recognized', img)
    cv.waitKey()
