from djitellopy import Tello
import cv2, math, time

tello = Tello()

tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

while True:
    img = tello.get_frame_read().frame
    cv2.imshow('stream', img)
    cv2.waitKey(1)


#for x in range (200):
#    dist = tello.get_distance_tof()
#    print(f"Distance {dist} cm. \n")

tello.end()
#tello.takeoff()

#tello.move_left(100)
#tello.rotate_clockwise(90)
#tello.move_forward(100)

#tello.land()
