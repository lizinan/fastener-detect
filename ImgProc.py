import cv2
import numpy as np

import os

def callback(x): pass

def showColorSpaces(img_path, window_name):
    scale = 4
    pics = []
    for filename in os.listdir(img_path):
        if os.path.splitext(filename)[1] != '.jpg': continue
        img = cv2.imread(img_path + "/" + filename)
        img_sz = img.shape
        img = cv2.resize(img, (int(img_sz[1]/scale), int(img_sz[0]/scale)))
        pics.append(img)

    cv2.namedWindow(window_name)
    cv2.createTrackbar("LH",window_name,0,255,callback)
    cv2.createTrackbar("LS",window_name,0,255,callback)
    cv2.createTrackbar("LV",window_name,0,255,callback)
    cv2.createTrackbar("UH",window_name,255,255,callback)
    cv2.createTrackbar("US",window_name,255,255,callback)
    cv2.createTrackbar("UV",window_name,255,255,callback)
    cv2.createTrackbar("no",window_name,0,len(pics)-1,callback)

    while True:

        pic_no = cv2.getTrackbarPos("no", window_name)
        img = pics[pic_no]

        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("LH",window_name)
        l_s = cv2.getTrackbarPos("LS",window_name)
        l_v = cv2.getTrackbarPos("LV",window_name)
        u_h = cv2.getTrackbarPos("UH",window_name)
        u_s = cv2.getTrackbarPos("US",window_name)
        u_v = cv2.getTrackbarPos("UV",window_name)
        
        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h,u_s,u_v])


        mask = cv2.inRange(hsv_img,l,u)
        res = cv2.bitwise_and(img, img ,mask=mask)

        imgs = np.hstack([img, res])
        cv2.imshow('mask', mask)
        cv2.imshow(window_name, imgs)
    
        key = cv2.waitKey(1)
        if key == "q": 
            break
    
    cv2.destroyAllWindows()


def testBinThresh(img_path, window_name):

    scale = 4
    pics = []
    for filename in os.listdir(img_path):
        if os.path.splitext(filename)[1] != '.jpg': continue
        img = cv2.imread(img_path + "/" + filename)
        img_sz = img.shape
        img = cv2.resize(img, (int(img_sz[1]/scale), int(img_sz[0]/scale)))
        pics.append(img)

    cv2.namedWindow(window_name)
    cv2.createTrackbar("g", window_name, 0, 255, callback)
    cv2.createTrackbar("arg1", window_name, 3, 30, callback)
    cv2.createTrackbar("arg2", window_name, 3, 30, callback)
    cv2.createTrackbar("no",window_name,0,len(pics)-1,callback)


    while True:
        pic_no = cv2.getTrackbarPos("no", window_name)


        img = pics[pic_no]

        if len(img.shape) == 2: 
            gray_img = img
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_val = cv2.getTrackbarPos("g",window_name)

        # _, binary_img = cv2.threshold(gray_img, gray_val, 255, cv2.THRESH_BINARY)


        arg1 =  cv2.getTrackbarPos("arg1", window_name)
        if arg1%2 == 0: arg1 += 1
        arg2 =  cv2.getTrackbarPos("arg2", window_name)
        binary_img = cv2.adaptiveThreshold(gray_img, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, arg1, arg2)

        imgs = np.hstack([gray_img, binary_img])
        cv2.imshow(window_name, imgs)
        
        key = cv2.waitKey(1)
        if key == "q": 
            break
    cv2.destroyAllWindows()
'''
function: test binary threshold for gray img using trackbar
'''
def findBinThresh(img, window_name):
    cv2.namedWindow(window_name)
    cv2.createTrackbar("g", window_name, 0, 255, callback)

    while True:
        if len(img.shape) == 2: 
            gray_img = img
        else:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_sp = gray_img.shape
        gray_img = cv2.resize(gray_img, (int(img_sp[1]/4), int(img_sp[0]/4)))   
        gray_val = cv2.getTrackbarPos("g",window_name)

        _, binary_img = cv2.threshold(gray_img, gray_val, 255, cv2.THRESH_BINARY)

        imgs = np.hstack([gray_img, binary_img])
        cv2.imshow(window_name, imgs)
        
        key = cv2.waitKey(1)
        if key == "q": 
            break
    cv2.destroyAllWindows()

n = 0
x1 = 0
y1 =0
def getColorValue(img_path, window_name):
    scale = 3
    pics = []
    for filename in os.listdir(img_path):
        if os.path.splitext(filename)[1] != '.jpg': continue
        img = cv2.imread(img_path + "/" + filename)
        img_sz = img.shape
        img = cv2.resize(img, (int(img_sz[1]/scale), int(img_sz[0]/scale)))
        pics.append(img)

    cv2.namedWindow(window_name)
    
    def mouseCallback(event,x,y,flags,param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        global n, x1, y1
        n = n + 1
        if n%2 == 1:
            x1 = x
            y1 = y
        else:
            x2 = x
            y2 = y
            sum = [0,0,0]
            for i in range(x2-x1):
                for j in range(y2-y1):
                    for k in range(3):
                        sum[k] += hsv_img[i][j][k]
            print(sum)
            area = (x2-x1)*(y2-y1)
            print("average: ", sum[0]/(area), " ", sum[1]/area, " ", sum[2]/area)

        # print("BGR value B:{}, G:{}, R:{}".format(img[x][y][0], img[x][y][1], img[x][y][2]))
        # print("HSV value H:{}, S:{}, V:{}".format(hsv_img[x][y][0], hsv_img[x][y][1], hsv_img[x][y][2]))

    cv2.setMouseCallback(window_name, mouseCallback)


    while True:
        pic_no = cv2.getTrackbarPos("no", window_name)

        img = pics[pic_no]
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        imgs = np.hstack([img, hsv_img])
        cv2.imshow(window_name, imgs)
        
        key = cv2.waitKey(1)
        if key == "q": 
            break
    cv2.destroyAllWindows()
'''
function: extract object regions using hough detection
return val: x1, x2, x3, x4
debug: show hough result 
'''
def extractRegions(img, debug = None):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    kernel_sz = 3
    kernel = np.ones((kernel_sz,kernel_sz), np.uint8)
    open_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE ,kernel)
    edges = cv2.Canny(open_img, threshold1=50, threshold2=200)
    # use hough detection
    lines = cv2.HoughLines(edges, rho=1, theta= np.pi , threshold=20)

    x = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        x.append(x1)
        cv2.line(gray, (x1, y1), (x2, y2), (255,0,255), 1)

    # extract object region
    x  = sorted(x)
    left_region = []
    right_region = []
    mid = int(img.shape[1]/2)
    for xi in x:
        if xi > mid / 4 and xi < mid:
            left_region.append(xi)
        if xi < mid * 7 / 4.5 and xi > mid:
            right_region.append(xi)

    if debug:
        print(left_region, right_region)
        if left_region != [] and  right_region!= []:
            cv2.line(gray, (left_region[0], 0), (left_region[0], img.shape[0] - 1), (255,255,255), 4)
            cv2.line(gray, (left_region[-1], 0), (left_region[-1], img.shape[0] - 1), (255,255,255), 4)
            cv2.line(gray, (right_region[0], 0), (right_region[0], img.shape[0] - 1), (255,255,255), 4)
            cv2.line(gray, (right_region[-1], 0), (right_region[-1], img.shape[0] - 1), (255,255,255), 4)
        
        imgs = np.hstack([binary,open_img])
        t = np.hstack([edges, gray])
        imgs = np.vstack([imgs, t])
        cv2.imshow("result", img)
        cv2.imshow("test", imgs)
        cv2.imshow("res", gray)


    return left_region[0], left_region[-1], right_region[0], right_region[-1] 

'''
function: create round kernel
'''
def createRoundKernel(kernel_sz):
    kernel = np.zeros((kernel_sz,kernel_sz), dtype = np.uint8)
    if kernel_sz%2 == 1:
        center_radius = int((kernel_sz-1)/2)
        cv2.circle(kernel, (center_radius, center_radius), center_radius, (0,0,0), -1, cv2.LINE_AA)
    else:
        center_radius = int(kernel_sz/2)
        cv2.circle(kernel, (center_radius, center_radius), center_radius, (0,0,0), -1, cv2.LINE_AA)
    return kernel

'''
function: create diamond kernel
return val: kernel
'''
def createDiamondKernel(kernel_sz):
    kernel = np.zeros((kernel_sz,kernel_sz), dtype = np.uint8)
    center = int((kernel_sz -1)/2)
    for i in range(kernel_sz):
        if i < center:
            kernel[i][center-i] = 1
            kernel[i][center+i] = 1
        else:
            kernel[i][3*center- i] = 1
            kernel[i][i - center] = 1

    return kernel

'''
function: extract object region using minAreaRect
return val: roi region
'''
def extractRegionByRect(img):

    img_cp = img.copy()
    x1, x2, x3, x4 = extractRegions(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray_img, 255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 5, 25)


    roi = 255*np.ones(binary.shape, dtype=np.uint8) - binary
    roi[:, 0:x1] = 0
    roi[:, x4:] = 0


    gau_img = cv2.GaussianBlur(roi, (3,3), 0.5)
    kernel_sz = 7
    kernel = createDiamondKernel(kernel_sz)
    morph_img = cv2.morphologyEx(gau_img, cv2.MORPH_CLOSE, kernel)

    dis_imgs = np.hstack([roi, gau_img, morph_img])
    cv2.imshow("fisrt process: roi, gau_img, morph_img", dis_imgs)

    cnts, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = []
    boxes = []
    areas = []
    for cnt in cnts:
        if len(cnt) < 100: continue

        rect = cv2.minAreaRect(cnt)
        

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        areas.append(rect[1][0]*rect[1][1])
        boxes.append([box[0][0], box[0][1], box[2][0], box[2][1]])
        print(box)


        cv2.drawContours(img_cp, [box], 0, (255,0,0), 2)
    cv2.imshow("img_cp", img_cp)
    
    max_index = areas.index(max(areas))
    areas[max_index] = 0
    sec_max_index = areas.index(max(areas))
    region.append(boxes[max_index])
    region.append(boxes[sec_max_index])
    print("region", region)
    
    return region
    

'''
function extract object features
input: color_img
'''
def extractFeature(img):

    region = extractRegionByRect(img)
    region = sorted(region)
    # crop regions
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_img = 255*np.ones(gray_img.shape, dtype=np.uint8)
    y_upper = max(region[0][1],region[1][1])
    y_lower = min(region[0][3],region[1][3])

    x_ext = 10
    y_ext = 10
    if y_upper > y_ext: y_upper -= y_ext
    if y_lower < gray_img.shape[0] - y_ext: y_lower += y_ext
    
    for reg in region:
        reg[0] -= x_ext
        reg[2] += x_ext
        new_img[y_upper: y_lower+1, reg[0]:reg[2]] = gray_img[y_upper: y_lower+1, reg[0]:reg[2]]
        cv2.rectangle(img, (reg[0], y_upper), (reg[2], y_lower), (0,255,0), 1)
        print("data", y_upper," ", y_lower+1," ", reg[0]," ", reg[2])

    _, binary = cv2.threshold(new_img, 80, 255, cv2.THRESH_BINARY)
    kernel_sz = 11
    kernel = createDiamondKernel(kernel_sz)
    morph_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((kernel_sz, kernel_sz))
    morph_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    dis_imgs = np.hstack([new_img, binary, morph_img])
    cv2.imshow("new_img, binary, morph_img", dis_imgs)
    cnts, _ = cv2.findContours(morph_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    left_cir = []
    right_cir = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 10: continue
        ((x,y),r) = cv2.minEnclosingCircle(cnt)
        if r > min(gray_img.shape[0], gray_img.shape[1]) / 16: continue
        cv2.circle(img, (int(x), int(y)), int(r), (255,0,0),2)
        
        if x < region[0][2]:
            left_cir.append([x,y,r])
        elif x > region[1][0]:
            right_cir.append([x,y,r])

    circles = []
    if len(left_cir) > 2:
        center = (region[0][0] + region[0][2])/2
        dis = [abs(cir[0] - center) for cir in left_cir]
        min_dis_index = dis.index(min(dis))
        dis[min_dis_index] = center
        sec_min_dis_index = dis.index(min(dis))

        circles.append(left_cir[min_dis_index])
        circles.append(left_cir[sec_min_dis_index])
    else:
        for cir in left_cir: circles.append(cir)

    if len(right_cir) > 2:
        center = (region[1][0] + region[1][2])/2
        dis = [abs(cir[0] - center) for cir in right_cir]
        min_dis_index = dis.index(min(dis))
        dis[min_dis_index] = center
        sec_min_dis_index = dis.index(min(dis))
        
        circles.append(right_cir[min_dis_index])
        circles.append(right_cir[sec_min_dis_index])
    else:
        for cir in right_cir: circles.append(cir)

    print(circles)
    for cir in circles:
        cv2.circle(img, (int(cir[0]), int(cir[1])), int(cir[2]), (255,255,0),2)




    cv2.imshow("rect img", img)
    # cv2.imshow("test", test_img)
    cv2.waitKey(0)

'''
funtion: get template
return val: left and right gray template
'''
def getTemplate(img):
    region = extractRegionByRect(img)
    region = sorted(region)
    # crop regions
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    y_upper = max(region[0][1],region[1][1])
    y_lower = min(region[0][3],region[1][3])

    # x_ext = 10
    y_ext = 10
    if y_upper > y_ext: y_upper -= y_ext
    if y_lower < gray_img.shape[0] - y_ext: y_lower += y_ext
    
    left_template = gray_img[y_upper: y_lower+1, region[0][0]:region[0][2]]
    right_template = gray_img[y_upper: y_lower+1, region[1][0]:region[1][2]]

    return left_template, right_template



def shapeMatch(img, template):
    left_template, right_template = getTemplate(template)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res_l = cv2.matchTemplate(gray_img, left_template, cv2.TM_CCOEFF_NORMED)
    res_r = cv2.matchTemplate(gray_img, right_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.55

    loc = np.where( res_l >= threshold)
    if loc[0].size == 0:
        print("left is abnormal")
    else:
        print("left is normal")
    w, h = left_template.shape[::-1]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
        break

    loc = np.where( res_r >= threshold)
    if loc[0].size == 0:
        print("right is abnormal")
    else:
        print("right is normal")
    w, h = right_template.shape[::-1]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)
        break

    t = np.hstack([left_template, right_template])
    cv2.imshow("left and right template", t)
    cv2.imshow("template match img",img)
    cv2.waitKey(0)


def combinePictures(img_path, save_path):
    filenames = os.listdir(img_path)
    filenames.sort(key = lambda x:int(x[0:-4]))
    img_idx = 1
    for i in range(0, len(filenames), 2):
        img1 = cv2.imread(img_path + "/" + filenames[i])
        img2 = cv2.imread(img_path + "/" + filenames[i+1])
        cv2.imwrite(save_path+"/"+str(img_idx)+".jpg", np.vstack([img1, img2]))
        img_idx += 1


def rmvBg(img_path, save_path):
    for filename in os.listdir(img_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(img_path + "/" + filename)
            [x1,x2,x3,x4] = extractRegions(img)
            img[:, 0:x1, :] = 0
            img[:,x2:x3, :] = 0
            img[:, x4:, :] = 0
            cv2.imwrite(save_path+"/"+filename, img)
    print("done")



if __name__ == "__main__":
    # img_path = "./data/data_rmv/val"
    # save_path =img_path
    # rmvBg(img_path, save_path)

    ### 合并img
    # img_path = "./val"
    # save_path = "./val"
    # combinePictures(img_path, save_path)



    # img_path = "./error"
    img_path = "./vid_pic"
    window_name = "show color spaces"
    # crop_l= (124, 245)
    # crop_r = (225, 431)
    # test = cv2.imread(img_path + "/" + "4.jpg")
    # test = cv2.resize(test, (640, 640))
    # extractRegions(test, True)
    # test = cv2.resize(test, (640, 640))
    # cv2.rectangle(test, crop_l, crop_r, (0,0,0), 2)
    
    # cv2.imshow("or", test)
    # test = test[crop_l[0]: crop_r[0], crop_l[1]:crop_r[1]]
    # test = cv2.resize(test, (640, 640))
    # # gx = cv2.Sobel(test, cv2.CV_32F, 1, 0, ksize=1)
    # # gy = cv2.Sobel(test, cv2.CV_32F, 0, 1, ksize=1)
    # # mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


    # ret2, dst2 = cv2.threshold(test, 0 , 255, cv2.adaptiveThreshold)
    # cv2.imshow("test", dst2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    showColorSpaces(img_path, window_name)

    # testBinThresh(img_path, window_name)

    # getColorValue(img_path, window_name )

    # img1 = cv2.imread(img_path + "/" + str(i) + ".jpg")
    # img2 = cv2.imread(img_path + "/" + str(i+1) + ".jpg")
    # img = np.vstack([img1, img2])
    # img_sz = img.shape
    # scale = 4
    # img = cv2.resize(img, (int(img_sz[1]/scale), int(img_sz[0]/scale)))   

    # img_path = "./pics/lose"
    # i = 49
    # img1 = cv2.imread(img_path + "/" + str(i) + ".jpg")
    # img2 = cv2.imread(img_path + "/" + str(i+1) + ".jpg")
    # template = np.vstack([img1, img2])
    # img_sz = template.shape
    # scale = 4
    # template = cv2.resize(template, (int(img_sz[1]/scale), int(img_sz[0]/scale)))   

    # shapeMatch(template, img)
    # extractFeature(img)


