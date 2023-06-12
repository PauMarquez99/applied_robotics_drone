import djitellopy as tello

tello = tello.Tello()
tello.connect()
tello.streamon()

print(tello.get_battery())

tello.streamoff()
tello.end()