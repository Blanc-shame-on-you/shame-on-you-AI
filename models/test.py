'''import db
import cv2
cur =db.setting()
img = db.load(cur)
cv2.imshow("2",img)'''
'''import cv2
import face_recognition

cap =cv2.VideoCapture(0)
while cap.isOpened():
    _,img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img)
    print(len(face_locations))

cap.release()'''

'''import db
import cv2
cursor,con=db.setting()
cap =cv2.VideoCapture(0)
while cap.isOpened():
    _,img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    db.test_save(img,cursor,con)
    cv2.imshow("test",db.load(cursor))
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()'''


import cv2
import db
cursor,_=db.setting()
imgs = db.load(cursor)
print(len(imgs))
print(imgs[1][0].shape)
cv2.imshow("test",imgs[(len(imgs)-1)][0])
cv2.waitKey(1)