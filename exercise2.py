import robomaster
from robomaster import robot

if __name__ == '__main__':
    tl_drone = robot.Drone()
    tl_drone.initialize()

    tl_flight = tl_drone.flight

    tl_battery = tl_drone.battery
    battery_info = tl_battery.get_battery()
    print('Drone battery soc: {0}'.format(battery_info))

    if battery_info > 10 :
        tl_flight.takeoff().wait_for_completed()
        tl_flight.up(distance = 50).wait_for_completed()

        tl_flight.forward(distance = 50).wait_for_completed()
        tl_flight.rotate(angle = 90).wait_for_completed()
        tl_flight.forward(distance = 50).wait_for_completed()
        tl_flight.rotate(angle = 90).wait_for_completed()
        tl_flight.forward(distance = 50).wait_for_completed()
        tl_flight.rotate(angle = 90).wait_for_completed()
        tl_flight.forward(distance = 50).wait_for_completed()
        tl_flight.rotate(angle = 90).wait_for_completed()

        tl_flight.down(distance = 50).wait_for_completed()
        tl_flight.land().wait_for_completed()

    tl_drone.close()