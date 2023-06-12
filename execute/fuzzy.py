import cv2, threading
from djitellopy import tello

############### GLOBAL VARIABLES ###############

tello = tello.Tello()
img = 0
proc_img = 0
face_center = [0,0]
face_area = 0
lr_y = [0, 0]
vel_ud = 0
vel_cf = 0
fuzzy = False
find_face = False
show_video = False
print_res_find_face = False
print_res_fuzzy = False

avg_area = [0] * 3
avg_lr = [0] * 3
avg_yaw = [0] * 3
avg_ud = [22000] * 3


############### CONSTANTS ###############

CENTER_IMG_X = 480

CENTER_IMG_Y = 360

INPUT_FIG_LR = [
    (["left",	"flat"],	    440, 	 330),
    (["left", 	"falling"], 	330, 	 110),
    (["med", 	"rising"], 	    330, 	 110),
    (["med", 	"flat"], 	    110, 	 -110),
    (["med", 	"falling"], 	-110, 	 -330),
    (["right", 	"rising"],		-110, 	 -330),
    (["right", 	"flat"],		-330, 	 -440)
]

INPUT_FIG_UD = [
    (["down",	"flat"],	    320, 	240),
    (["down", 	"falling"], 	240, 	80),
    (["med", 	"rising"], 	    240, 	80),
    (["med", 	"flat"], 	    80, 	-80),
    (["med", 	"falling"], 	-80, 	-240),
    (["up", 	"rising"],		-80, 	-240),
    (["up", 	"flat"],		-240, 	-320)
]

INPUT_FIG_CF = [
    (["close",	"flat"],	    2000, 	11000),
    (["close", 	"falling"], 	11000, 	16500),
    (["med", 	"rising"], 	    11000, 	16500),
    (["med", 	"flat"], 	    16500, 	33500),
    (["med", 	"falling"], 	33500, 	50000),
    (["far", 	"rising"],		33500, 	50000),
    (["far", 	"flat"],		50000, 220000)
]

RULES_LR = {
    "left":  -30,
    "med":    0,
    "right":  30,
}

RULES_UD = {
    "down":  50,
    "med":   0,
    "up":   -50,
}

RULES_CF = {
    "close":  42,
    "med":    0,
    "far":   -42,
}


############### FUNCTIONS ###############

def findFace():
    global proc_img
    global face_center
    global face_area
    global fuzzy
    global show_video
    global print_res_find_face

    if find_face:
        faceCascade = cv2.CascadeClassifier("resources/face_tracking/haarcascades/haarcascade_frontalface_default.xml")
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.2, 7)

        faces_center_list = []
        faces_area_list = []

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
            faces_center_list.append([x+w//2,y+h//2])
            faces_area_list.append(w*h)
            cv2.circle(img, (x+w//2,y+h//2), 5, (0,255,0), cv2.FILLED)
        
        proc_img = img

        if len(faces_area_list) > 0:
            i = faces_area_list.index(max(faces_area_list))
            face_center = faces_center_list[i]
            face_area = faces_area_list[i]
        else:
            face_center = [0,0]
            face_area = 0

        fuzzy = True
        show_video = True
        print_res_find_face = True

def getFigPosLRUD(x, input_fig):
    res = []
    for descrip in input_fig:
        if descrip[1] > x >= descrip[2]:
            res.append(descrip)
    return res

def getFigPosCF(x, input_fig):
    res = []
    for descrip in input_fig:
        if descrip[1] < x <= descrip[2]:
            res.append(descrip)
    return res

def calcMembershipVal(limits, x, isRisingBool):
    if isRisingBool:
        a = limits[0]
        b = limits[1]
        return (x-a)/(b-a)
    else:
        d = limits[1]
        c = limits[0]
        return (d-x)/(d-c)

def getMembershipValues(descriptions, x):
    res = []
    for description in descriptions:
        if description[0][1] == "flat":
            res.append([description[0][0], 1])
        elif description[0][1] == "rising":
            res.append([description[0][0], calcMembershipVal([description[1], description[2]], x, True)])
        elif description[0][1] == "falling":
            res.append([description[0][0], calcMembershipVal([description[1], description[2]], x, False)])
    return res

def getFiredRules(membership_values, rules):
    outputCenters = []
    firingStrenghts = []
    for m_val in membership_values:
        rule = m_val[0]
        firingStrenghts.append(m_val[1])
        outputCenters.append(rules[rule])
    return [outputCenters, firingStrenghts]

def defuzzifyLR(info):
    output_centers = info[0]
    firing_strengths = info[1]
    vel_lr = 0
    vel_yaw = 0
    for i, o_c in enumerate(output_centers):
        vel_lr += o_c * firing_strengths[i]
        vel_yaw += o_c * firing_strengths[i]
    total_memb_val = sum(firing_strengths)
    if total_memb_val > 0:
        return vel_lr, vel_yaw
    else:
        return 0, 0

def defuzzify(info):
    output_centers = info[0]
    firing_strengths = info[1]
    vel = 0
    for i, o_c in enumerate(output_centers):
        vel += o_c * firing_strengths[i]
    total_memb_val = sum(firing_strengths)
    if total_memb_val > 0:
        return vel
    else:
        return 0

def fuzzify():
    global lr_y
    global vel_ud
    global vel_cf
    global print_res_fuzzy

    if fuzzy:
        x = CENTER_IMG_X-face_center[0]
        y = CENTER_IMG_Y-face_center[1]

        descriptions_LR = getFigPosLRUD(x, INPUT_FIG_LR)
        membership_values_LR = getMembershipValues(descriptions_LR, x)
        info_LR = getFiredRules(membership_values_LR, RULES_LR)

        descriptions_UD = getFigPosLRUD(y, INPUT_FIG_UD)
        membership_values_UD = getMembershipValues(descriptions_UD, y)
        info_UD = getFiredRules(membership_values_UD, RULES_UD)

        descriptions_CF = getFigPosCF(face_area, INPUT_FIG_CF)
        membership_values_CF = getMembershipValues(descriptions_CF, face_area)
        info_CF = getFiredRules(membership_values_CF, RULES_CF)
        
        lr_y = defuzzifyLR(info_LR)
        vel_ud = defuzzify(info_UD)
        vel_cf = defuzzify(info_CF)
    
    print_res_fuzzy = True
    
def getVideo():
    global img
    global find_face

    img = img_read.frame
    (h, w) = img.shape[:2]
    cv2.circle(img, (w//2, h//2), 7, (255, 255, 255), -1)

    find_face = True

def getAvgArea(area):
    global avg_area

    avg_area = avg_area[-1:] +  avg_area[:-1]
    avg_area[0] = area
    return sum(avg_area) / len(avg_area)

def getAvgLR(lr):
    global avg_lr

    avg_lr = avg_lr[-1:] +  avg_lr[:-1]
    avg_lr[0] = lr
    return sum(avg_lr) / len(avg_lr)

def getAvgUD(ud):
    global avg_ud

    avg_ud = avg_ud[-1:] +  avg_ud[:-1]
    avg_ud[0] = ud
    return sum(avg_ud) / len(avg_ud)

def getAvgYaw(y):
    global avg_yaw

    avg_yaw = avg_yaw[-1:] +  avg_yaw[:-1]
    avg_yaw[0] = y
    return sum(avg_yaw) / len(avg_yaw)

############### MAIN ###############

tello.connect()
tello.set_video_fps(tello.FPS_30)
tello.streamon()
img_read = tello.get_frame_read()
tello.takeoff()

while True:
    t_get_video = threading.Thread(target=getVideo, args=())
    t_find_face = threading.Thread(target=findFace, args=())
    t_fuzzify = threading.Thread(target=fuzzify, args=())

    if print_res_fuzzy and print_res_find_face:
        lr =  int(getAvgLR(lr_y[0]))
        cf =  int(getAvgArea(vel_cf))
        ud =  int(getAvgUD(vel_ud))
        yaw = int(getAvgYaw(lr_y[1]))

        print("\nD_X: {} \t D_Y: {} \t A: {}".format(CENTER_IMG_X-face_center[0], CENTER_IMG_Y-face_center[1], face_area))
        print("V_LR: {} - V_CF: {} - V_UD: {} - V_Y: {}".format(lr, cf, ud, yaw))
        tello.send_rc_control(lr, cf, ud, yaw)

    t_get_video.start()
    t_find_face.start()
    t_fuzzify.start()

    t_get_video.join()
    t_find_face.join()
    t_fuzzify.join()

    if show_video:
        cv2.imshow('stream', proc_img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
tello.land()
tello.streamoff()
tello.end()
