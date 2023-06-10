from djitellopy import Tello
import cv2, math, time


def findFace(img):
    faceCascade = cv2.CascadeClassifier("codes/face_tracking/haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    faces_center_list = []
    faces_area_list = []

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        faces_center_list.append([x+w//2,y+h//2])
        faces_area_list.append(w*h)
        cv2.circle(img, (x+w//2,y+h//2), 5, (0,255,0), cv2.FILLED)
    
    if len(faces_area_list) > 0:
        i = faces_area_list.index(max(faces_area_list))
        return img, [faces_center_list, faces_area_list[i]]
    else:
        return img, [[0,0], 0]

def closeness(distance):
    if distance > 70000:
        move = "forward"
        #tello.move_forward(10)
    elif distance < 58500:
        move = "backwards"
        #tello.move_back(10)
    else:
        move = "nada"
    return move

tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    img = tello.get_frame_read().frame
    proc_img, face_info = findFace(img)
    move = closeness(face_info[1])
    print("Center: {} - Area: {} - Move: {}".format(face_info[0], face_info[1], move))
    cv2.imshow('stream', proc_img)


tello.end()
cv2.destroyAllWindows()
#tello.land()