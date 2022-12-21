import os
import cv2
import numpy as np


def createVid(path, fps, vid_name):
    # path = './vid_pic' 
    # path = './images_time_1830_left'

    fps = 10
    filenames = os.listdir(path)
    filenames.sort(key = lambda x:int(x[0:-4]))
    test = path + "/" + filenames[0]
    img = cv2.imread(test)
    img_sp = img.shape[0:2]
    # vid_name = "3.mp4"
    # video = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, img_sp)
    video = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_sp[1], img_sp[0]))

    img_names = os.listdir(path)
    img_names.sort(key = lambda x:int(x[0:-4]))
    for img_name in img_names:
        img = cv2.imread(path + "/" +img_name)
        # img = cv2.transpose(img)
        if img is None:
            print("error")
            continue
        # cv2.imshow("i", img)
        # cv2.waitKey(1)
        video.write(img)
    
    print("done")
    video.release()
    cv2.destroyAllWindows()

    # return vid_name

def playVid(vid_name):
    cap = cv2.VideoCapture(vid_name)

    while cap.isOpened():
        ret, frame=cap.read()

        if ret == False:
            print("error")
            break
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    path = "./vid_pic"
    rate = 10
    vid_name = "test.mp4"
    createVid(path, rate, vid_name)
    # vid_name = "1.mp4"
    # playVid(vid_name)