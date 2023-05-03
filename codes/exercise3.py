import robomaster
from robomaster import robot
import time
import cv2 as cv2
from threading import Thread

def recordVideo(status):
    while status:
        img = tl_camera.read_cv2_image()
        cv2.imshow('Drone', img)
        cv2.waitKey(1)
        time.sleep(1)
    
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()

if __name__ == '__main__':
    tl_drone = robot.Drone()
    tl_drone.initialize()

    tl_flight = tl_drone.flight

    tl_battery = tl_drone.battery
    battery_info = tl_battery.get_battery()
    print('Drone battery soc: {0}'.format(battery_info))

    tl_camera = tl_drone.camera
    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps('high')
    tl_camera.set_resolution('high')
    tl_camera.set_bitrate(6)

    if battery_info > 10 :
        tl_flight.takeoff().wait_for_completed()
        
        tl_flight.rc(a=0, b=30, c=0, d=50)
        time.sleep(10)
        tl_flight.rc(a=0, b=0, c=0, d=0)
        time.sleep(4)

        tl_flight.land().wait_for_completed()

    tl_drone.close()