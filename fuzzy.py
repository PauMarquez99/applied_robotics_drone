import cv2, threading


############### GLOBAL VARIABLES ###############

vid = cv2.VideoCapture(1)
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


############### CONSTANTS ###############

CENTER_IMG_X = 960

CENTER_IMG_Y = 540

INPUT_FIG_LR = [
    (["left",	"flat"],	    -800, 	-400),
    (["left", 	"falling"], 	-400, 	0),
    (["med", 	"rising"], 	    -400, 	0),
    (["med", 	"falling"], 	0, 	    400),
    (["right", 	"rising"],		0, 	    400),
    (["right", 	"flat"],		400, 	800)
]

INPUT_FIG_UD = [
    (["down",	"flat"],	    -400, 	-200),
    (["down", 	"falling"], 	-200, 	0),
    (["med", 	"rising"], 	    -200, 	0),
    (["med", 	"falling"], 	0, 	    200),
    (["up", 	"rising"],		0, 	    200),
    (["up", 	"flat"],		200, 	400)
]

INPUT_FIG_CF = [
    (["close",	"flat"],	    20000, 	50000),
    (["close", 	"falling"], 	50000, 	70000),
    (["med", 	"rising"], 	    50000, 	70000),
    (["med", 	"falling"], 	70000, 	200000),
    (["far", 	"rising"],		70000, 	200000),
    (["far", 	"flat"],		200000, 700000)
]

RULES_LR = {
    "left":  -66,
    "med":    0,
    "right":  66,
}

RULES_UD = {
    "down":  66,
    "med":   0,
    "up":   -66,
}

RULES_CF = {
    "close": -66,
    "med":    0,
    "far":    66,
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

def getFigPos(x, input_fig):
    res = []
    for descrip in input_fig:
        if x < descrip[2] and x >= descrip[1]:
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
    vel_lf = 0
    vel_yaw = 0
    for i, o_c in enumerate(output_centers):
        vel_lf += o_c * firing_strengths[i]
        vel_yaw += o_c * firing_strengths[i]
    total_memb_val = sum(firing_strengths)
    if total_memb_val > 0:
        return vel_lf, vel_yaw
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
        a = face_area

        descriptions_LR = getFigPos(x, INPUT_FIG_LR)
        membership_values_LR = getMembershipValues(descriptions_LR, x)
        info_LR = getFiredRules(membership_values_LR, RULES_LR)

        descriptions_UD = getFigPos(y, INPUT_FIG_UD)
        membership_values_UD = getMembershipValues(descriptions_UD, y)
        info_UD = getFiredRules(membership_values_UD, RULES_UD)

        descriptions_CF = getFigPos(a, INPUT_FIG_CF)
        membership_values_CF = getMembershipValues(descriptions_CF, a)
        info_CF = getFiredRules(membership_values_CF, RULES_CF)
        
        lr_y = defuzzifyLR(info_LR)
        vel_ud = defuzzify(info_UD)
        vel_cf = defuzzify(info_CF)
    
    print_res_fuzzy = True
    
def getVideo():
    global img
    global find_face

    _, img = vid.read()
    (h, w) = img.shape[:2]
    cv2.circle(img, (w//2, h//2), 7, (255, 255, 255), -1)

    find_face = True


############### MAIN ###############

while True:
    t_get_video = threading.Thread(target=getVideo, args=())
    t_find_face = threading.Thread(target=findFace, args=())
    t_fuzzify = threading.Thread(target=fuzzify, args=())

    if print_res_fuzzy and print_res_find_face:
        print("D_X: {} \t D_Y: {} \t A: {}\n".format(CENTER_IMG_X-face_center[0], CENTER_IMG_Y-face_center[1], face_area))
        print("V_LF: {} - V_Y: {} - V_UD: {} - V_CF: {}".format(int(lr_y[0]), int(lr_y[1]), int(vel_ud), int(vel_cf)))

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
