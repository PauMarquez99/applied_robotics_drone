from djitellopy import tello
import cv2, math, time

upper_limit = 15000
lower_limit = 10000
area_list = [(upper_limit + lower_limit)/2] * 3

def FindFace(img):
    faceCascade = cv2.CascadeClassifier("codes/face_tracking/haarcascades/haarcascade_frontalface_default.xml")
    #imgRGB = cv2.cvtColor(img,cv2.COLOR_YUV2RGB)
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

def GetAreaAvg(curr_area):
    global area_list
    area_list = area_list[-1:] +  area_list[:-1]
    area_list[0] = curr_area
    return sum(area_list) / len(area_list)

tello = tello.Tello()
tello.connect()
tello.set_video_fps(tello.FPS_30)
tello.streamon()

if tello.get_battery() > 20:
    img_read = tello.get_frame_read()
    # tello.takeoff()

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        img = img_read.frame
        (h, w) = img.shape[:2]
        # print("x: {} - y: {}".format(w//2, h//2))

        proc_img, face_info = FindFace(img)
        # print(face_info[0])
        print(face_info[1])
        # # move = Closeness(area_avg)
        # print("Center: {} - Area: {} - Average {}".format(face_info[0], face_info[1], area_avg))
        cv2.imshow('stream', img)

    cv2.destroyAllWindows()

# tello.land()
# tello.end()