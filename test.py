from djitellopy import Tello
import cv2, math, time


def findFace(img):
    faceCascade = cv2.CascadeClassifier("codes/face_tracking/haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 7)

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

tello = Tello()

tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

while True:
    img = tello.get_frame_read().frame
    proc_img, face_info = findFace(img)
    print("Center: {} - Area: {}".format(face_info[0], face_info[1]))
    cv2.imshow('stream', proc_img)
    cv2.waitKey(1)

#for x in range (200):
#    dist = tello.get_distance_tof()
#    print(f"Distance {dist} cm. \n")p

tello.end()
#tello.takeoff()

#tello.move_left(100)
#tello.rotate_clockwise(90)
#tello.move_forward(100)

#tello.land()
