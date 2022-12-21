import cv2
import detect
import os
import time
import numpy as np
'''
function: translate yolo bounding boxes to loss class
input: list of list [[class, ]]
output: fallback class
'''
def initYolo(weights, conf):
    return detect.DetectAPI(weights = weights, conf_thres=conf)

def average(img, crop_l, crop_r):
    if len(img.shape) == 2:
        sum = 0
        crop = img[crop_l[1]:crop_r[1], crop_l[0]:crop_r[0]]
        sum = np.sum(np.reshape(crop, (crop.size, )))

        area = (crop_r[0] - crop_l[0])*(crop_r[1] - crop_l[1])
        sum = sum/area

    else:
        sum = [0,0,0]
        w = crop_r[0] - crop_l[0]
        h = crop_r[1] - crop_l[1]
        for i in range(w):
            for j in range(h):
                val = img[crop_l[0]+i][crop_l[1]+j]
                # print(val)
                for k in range(3):
                    sum[k] += img[crop_l[0]+j][crop_l[1]+i][k]

        area = w*h
        # print("sum", sum)
        for k in range(3):
            sum[k] = sum[k]/area

        # print("average: ", sum)
    return sum


def compare(s_crop, s_steel, s_insulate):
    print(type(s_crop))
    if isinstance(s_crop, np.float64):
        dis1 = (s_crop - 255)**2
        dis2 = (s_crop - s_insulate)**2
        print("here")
        print(s_crop," ", s_steel," ",s_insulate)
    else:
        # dis1 = (s_crop[0]-s_steel[0])**2
        # dis2 = (s_crop[0]- s_insulate[0])**2
        dis1 = (s_crop[0]-s_steel[0])*(s_crop[0]-s_steel[0]) + (s_crop[1]-s_steel[1])*(s_crop[1]-s_steel[1]) + (s_crop[2]-s_steel[2])*(s_crop[1]-s_steel[1])
        dis2 = (s_crop[0]-s_insulate[0])*(s_crop[0]-s_insulate[0]) + (s_crop[1]-s_insulate[1])*(s_crop[1]-s_insulate[1]) + (s_crop[2]-s_insulate[2])*(s_crop[2]-s_insulate[2])
    # print("crop and steel: ", dis1)
    # print("crop and insulate: ", dis2)
        t1 = s_crop[0]**2 +s_crop[1]**2 +s_crop[2]**2
        t2 = s_steel[0]**2+s_steel[1]**2+s_steel[2]**2
        t3 = s_insulate[0]**2+s_insulate[1]**2+s_insulate[2]**2
        print(s_crop," ", s_steel," ",s_insulate)
        print(t1," ", t2," ",t3)
    return True if dis1 > dis2 else False

def detectInsulate(im0, pos):

    # crop_l = (pos[2]-10,  pos[1]+int((pos[3]-pos[1])*0.3)+10)
    # crop_r = (pos[2]+30, crop_l[1] + 220)
    crop_l = (pos[2] , int( pos[1] + (pos[3] - pos[1])*0.35)+5)
    crop_r = (pos[2]+10, crop_l[1] + 140)

    # steel_l = (int( pos[2] - (pos[2]-pos[0])/15 ) , pos[3])
    # steel_r = (int( pos[2] + (pos[2]-pos[0])/15 ), int( pos[3] + 1/5*(pos[3] - pos[1]) ) )

    # insulate_l = (int( pos[2] - (pos[2]-pos[0])/3 ) , int( pos[1] + (pos[3] - pos[1])*3/10 ))
    # insulate_r = (int( pos[2] - (pos[2]-pos[0])/5 ) , int( pos[1] + (pos[3] - pos[1])/3 ))



    crop = im0[crop_l[1]:crop_r[1],crop_l[0]:crop_r[0]]
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    l = np.array([0, 67, 0])
    h = np.array([255, 255, 138])
    mask = cv2.inRange(crop, l, h)

    area = np.reshape(mask, (mask.size, ))
    s = np.sum(area>0)
    if s < area.size*0.43:
        return True




    # crop = im0[crop_l[0]:crop_r[0],crop_l[1]:crop_r[1]]
    # count = 0
    # for pixel in crop[0]:
    #     # print(pixel)
    #     if pixel[0] <= 30 and pixel[1] <= 30 and pixel[2] <=30:
    #         count+=1
    # print(count)
    # if count > 50:
    #     return True
    # # im1 = cv2.cvtColor(im0, cv2.COLOR_RGB2Lab)
    # # hsv_l = np.array([0, 0, 0])
    # # hsv_h = np.array([255,255,135])
    # # mask = cv2.inRange(im1,hsv_l, hsv_h)
    # mask = im0

    # s1 = average(mask, crop_l, crop_r)
    # s2 = average(mask, steel_l, steel_r)
    # s3 = average(mask, insulate_l, insulate_r)

    # cv2.rectangle(im0, crop_l, crop_r, (0,0,255), 3)
    # cv2.rectangle(im0, steel_l, steel_r, (0,255,0), 3)
    # cv2.rectangle(im0, insulate_l,insulate_r, (0,0,255), 3)

    # if compare(s1, s2, s3):
    #     # print("OK") 
    #     return True
    # # print(s1,',',s2,',',s3)
    # # if s1  < 220:
    # #     return True



    return False

def newDetectInsulate(im0):
    # gray_img = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    # binary_img = cv2.adaptiveThreshold(gray_img, 255,
    #                         cv2.ADAPTIVE_THRESH_MEAN_C,
    #                         cv2.THRESH_BINARY, 29,20)
    labImg = cv2.cvtColor(im0, cv2.COLOR_RGB2Lab)
    l = np.array([0, 0, 0])
    h = np.array([255,255,135])
    mask = cv2.inRange(labImg, l, h)


    thres = np.sum(mask > 0)
    print(thres, ",", mask.size)
    if thres < mask.size/2:
        print("true")
        return True
        
    else:
        return False



def processRlt2Qt(yolo_detector, img):

    st_time = time.time()
    im0, bounding_boxes = yolo_detector.detect([img])
    
    w = im0.shape[1]

    #get string and nut loc
    l_nut, l_string, l_insulate, r_nut, r_string, r_insulate = False, False, False, False, False, True
    for cls, pos, conf in bounding_boxes:
        center_pos = ( (pos[0]+pos[2])/2, (pos[1]+pos[3])/2)
        if center_pos[0] < w/2:
            if cls == 1: l_nut = True
            else: 
                l_string = True
                l_insulate = detectInsulate(im0, pos)
                # l = (pos[2]-10,  pos[1]+int((pos[3]-pos[1])*0.3)+10)
                # r = (pos[2]+30, l[1] + 220)
                # crop = img[l[0]:r[0], l[1]:r[1]]
                # l_insulate = newDetectInsulate(crop)
                # cv2.rectangle(im0, l, r, (0,255,0), 2)

        elif center_pos[0] > w/2:
            if cls == 1: r_nut = True
            else: r_string = True

    ed_time = time.time()

    rst = {0:l_string,
    1: l_nut,
    2: l_insulate,
    3: r_string,
    4: r_nut,
    5: r_insulate,
    }
    fps = round(1/(ed_time -  st_time))
    return im0, rst, fps





def processRlt(yolo_detector, img):

    st_time = time.time()
    im0, bounding_boxes = yolo_detector.detect([img])
    
    w = im0.shape[1]
    # h = im0.shape[0]
    # print(bounding_boxes)
    


    #get string and nut loc
    l_nut, l_string, l_insulate, r_nut, r_string, r_insulate = False, False, False, False, False, True
    for cls, pos, conf in bounding_boxes:
        center_pos = ( (pos[0]+pos[2])/2, (pos[1]+pos[3])/2)
        if center_pos[0] < w/2:
            if cls == 1: l_nut = True
            else: 
                l_string = True
                # crop = im0[pos[0]:pos[2], pos[1]:pos[3]]
                l_insulate = newDetectInsulate(img, pos)
        elif center_pos[0] > w/2:
            if cls == 1: r_nut = True
            else: r_string = True

    ed_time = time.time()

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    
    org = [(50,50), (50, 100), (50, 150),(400,50), (400, 100), (400, 150)]
    d = {True: "OK", False: "Lose"}
    im0 = cv2.putText(im0, " left nut: {}".format(d[l_nut]), org[0], font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    im0 = cv2.putText(im0, " left string: {}".format(d[l_string]), org[1], font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    im0 = cv2.putText(im0, " left insulate: {}".format(d[l_insulate]), org[2], font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    im0 = cv2.putText(im0, " right nut: {}".format(d[r_nut]), org[3], font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    im0 = cv2.putText(im0, " right string: {}".format(d[r_string]), org[4], font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    im0 = cv2.putText(im0, " right insulate: {}".format(d[r_insulate]), org[5], font, 
                    fontScale, color, thickness, cv2.LINE_AA)
 
    im0 = cv2.putText(im0, " fps: {}".format(1/(ed_time -  st_time)), (50, 200), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
 
    # while 1:

    #     cv2.imshow("test", im0)
    #     if cv2.waitKey(1) == 'q':
    #         break

    # cv2.destroyAllWindows()
    

    return im0


def detetctDrawback(yolo_detector, img_path, vid_name):
    img_names = os.listdir(img_path)
    img_names.sort(key = lambda x:int(x[0:-4]))

    fps = 10
    test = img_path + "/" + img_names[0]
    img = cv2.imread(test)
    img_sp = img.shape[0:2]
    video = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_sp[1], img_sp[0]))


    for img_name in os.listdir(img_path):
        img = cv2.imread(img_path + "/" + img_name)
        im0 = processRlt(yolo_detector, img)
        video.write(im0)

    video.release()
    print("done")


def test(yolo_detector,img):
    pics = []
    for filename in os.listdir(img_path):
        if os.path.splitext(filename)[1] != '.jpg': continue
        img = cv2.imread(img_path + "/" + filename)
        pics.append(img)
    window_name = "test"
    def callback(x): pass
    cv2.namedWindow(window_name)
    cv2.createTrackbar("no",window_name,0,len(pics)-1,callback)
    pic_no = 0
    while True:
        pic_no = cv2.getTrackbarPos("no", window_name)
        img = pics[pic_no]
        im0, bounding_boxes = yolo_detector.detect([img])
        w = im0.shape[1]

        pos = []
        for cls, posi, conf in bounding_boxes:
            center_pos = ( (posi[0]+posi[2])/2, (posi[1]+posi[3])/2)
            if center_pos[0] < w/2:
                if cls == 0:
                    pos = posi
                    break
        if pos == []:
            continue
        crop_l = (pos[2] , int( pos[1] + (pos[3] - pos[1])*0.35)+5)
        
        crop_r = (pos[2]+20, crop_l[1] + 140)
        crop = img[crop_l[1]:crop_r[1],crop_l[0]:crop_r[0]]
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        l = np.array([0, 67, 0])
        h = np.array([255, 255, 138])
        mask = cv2.inRange(crop, l, h)
        cv2.imshow("tt",mask)

        res = cv2.bitwise_and(crop,crop ,mask=mask)
        tt = np.reshape(mask, (mask.size, ))
        ss = np.sum(tt>0)
        print(tt.size,",",ss)
        if ss < tt.size/2:
            print("T")
        else:
            print("F")
        # count = 0
        # for pixel in crop[0]:
        #     # print(pixel)
        #     if pixel[0] <= 30 and pixel[1] <= 30 and pixel[2] <=30:
        #         count+=1
        # if count > 20:
        #     print("True")
        # else:
        #     print("False")
        
        

        # steel_l = (int( pos[2] - (pos[2]-pos[0])/15 ) , pos[3])
        # steel_r = (int( pos[2] + (pos[2]-pos[0])/15 ), int( pos[3] + 1/5*(pos[3] - pos[1]) ) )

        # insulate_l = (int( pos[2] - (pos[2]-pos[0])/3 ) , int( pos[1] + (pos[3] - pos[1])*3/10 ))
        # insulate_r = (int( pos[2] - (pos[2]-pos[0])/5 ) , int( pos[1] + (pos[3] - pos[1])/3 ))



        cv2.rectangle(im0, crop_l, crop_r, (0,255,0), 2)
        # cv2.rectangle(img, steel_l, steel_r, (0,255,0), 2)
        # cv2.rectangle(img, insulate_l, insulate_r, (0,255,0), 2)
    
        lab_im = cv2.resize(im0, (640, 640))
        
        mask = cv2.resize(crop, (crop.shape[1]*4, crop.shape[0]*4))
        cv2.imshow("la", lab_im)
        cv2.imshow("ma", mask)
        key = cv2.waitKey(1)
        if key == "q": 
            break
    cv2.destroyAllWindows()


'''
function: play vid with trackbar and see result
'''
def playVid(vid_path):
    pass

if __name__ == "__main__":


    #img_path = "./images_time_1830_left"
    img_path = "./vid_pic"
    weights = "./runs/train/exp9_weight/weights/best.pt"
    conf = 0.5
    vid_name = "detect.mp4"

    # init yolo detector
    yolo_detector = initYolo(weights, conf)

    # detectDrawbacks
    # img_path = "./images_time_1830_left"
    # t = cv2.imread(img_path + "/" + "4.jpg")
    test(yolo_detector, img_path)
    # processRlt(yolo_detector, test)


    
    # detetctDrawback(yolo_detector, img_path, vid_name)
