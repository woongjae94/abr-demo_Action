#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import cv2
import copy
import numpy as np
import math
import os

import time
import argparse

from crop_frames import CropFrames
from TFModel_mtsi3d import TFModel

import requests
import socket

from darknet.python.darknet import *

def sampling_frames(input_frames, sampling_num):
    total_num = len(input_frames)
    out_frames = []
    interval = 1
    if len(input_frames) > sampling_num:
        interval = math.floor(float(total_num) / sampling_num)

    print("sampling interval : {}".format(interval))
    interval = int(interval)
    if interval > 200:
        return out_frames

    for n in range(min(len(input_frames), sampling_num)):
        out_frames.append(input_frames[n*interval])

    # padding
    if len(out_frames) < sampling_num:
        print("before padding : {}".format(len(out_frames)))
        for k in range(sampling_num - len(out_frames)):
            out_frames.append(input_frames[-1])

    return out_frames

def pred_action(frames):
    result, confidence, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence > 0.7 and result != 'Doing other things':
        return result, confidence, top_3
    else:
        result = None

    return result, confidence, top_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--caption_video_length', type=int, default=64)
    parser.add_argument('--action_video_length', type=int, default=16)
    parser.add_argument('--action_thresh', type=int, default=20)
    parser.add_argument('--frame_thresh', type=int, default=10)
    parser.add_argument('--frame_diff_thresh', type=int, default=0.4)
    parser.add_argument('--waiting_time', type=int, default=8)

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60
    parser.add_argument('--port', type=str, default="8090")

    args = parser.parse_args()

    action_model = TFModel()

    ############# added by WoongJae ###################
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 0))
    ip = s.getsockname()[0]
    print("ip : ", ip, " ,port : ", args.port)
    mills = lambda: int(round(time.time() * 1000))
    cam_address = 'http://' + 'cam_container' + ':' + args.port + '/?action=stream'
    cwd_path = os.getcwd()
    ###################################################

    #cap = cv2.VideoCapture(args.cam)
    
    cap = cv2.VideoCapture(cam_address) # fix
    cap.set(3, args.width)
    cap.set(4, args.height)
    cap.set(5, args.fps)

    frames = []
    sampled_frames = []
    frame_num = 1
    start_frame = 1
    action_end_frame = 1
    motion_detect = False
    result = None
    action_list = []
    intent_list = []
    intent_result = ''

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while cap.isOpened():

        action_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        display_frame = copy.deepcopy(frame)
        display_frame = cv2.resize(display_frame, (224, 224))
        
        frame = cv2.resize(frame, (224, 224))

        frames.append(frame)

        # detect
        r = np_detect(yolo, meta, frame)

        if len(r) >= 1:

            if len(frames) >= 5:
                c_frame = frames[-1]
                c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
                b_frame = frames[-2]
                b_frame = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)
                a_frame = frames[-3]
                a_frame = cv2.cvtColor(a_frame, cv2.COLOR_BGR2GRAY)

                cb_frame_diff = cv2.absdiff(c_frame, b_frame)
                ba_frame_diff = cv2.absdiff(b_frame, a_frame)

                cba_frame_diff = cv2.absdiff(cb_frame_diff, ba_frame_diff)
                _, cba_frame_diff = cv2.threshold(cba_frame_diff, 30, 255, cv2.THRESH_BINARY)

                cb_diff_mask = np.array(cb_frame_diff > 10, dtype=np.int32)
                ba_diff_mask = np.array(ba_frame_diff > 10, dtype=np.int32)
                cba_diff_mask = np.array(cba_frame_diff > 10, dtype=np.int32)

                try:
                     diff_thresh = float(1.0*np.sum(cba_diff_mask)/max(np.sum(cb_diff_mask), np.sum(ba_diff_mask)))

                except:
                    diff_thresh = 0
                # print('(threshold : {})'.format(diff_thresh))

                if diff_thresh >= args.frame_diff_thresh and not motion_detect:
                    #start_frame = frame_num - 2
                    motion_detect = True
                    #print("start frame : {}\n".format(start_frame))
                # elif diff_thresh < 0.3 :
                #     motion_detect = False
                #     # sampled_frames = []

                if motion_detect:
                    #cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1)    # 12
                    # 모션 감지 flask로 보내기
                    # print(start_frame)

                    action_time = time.time()   # reset action_time

                    # when the movement stops
                    if diff_thresh < 0.1:# args.frame_diff_thresh:#frame_num >= start_frame + args.action_video_length:

                        if len(frames) >= args.frame_thresh:

                            #print('total frames : {}'.format(len(frames[start_frame:])))
                            sampled_frames = sampling_frames(frames, args.action_video_length)
                            if len(sampled_frames)==0:
                                frames=[]
                                motion_detect = False
                                continue

                            """
                            save_frames = cwd_path + "/Videos/test-video_{}_{}".format('mtsi3d-cropped', '4thdemo')
                            for num, f in enumerate(sampled_frames):
                                cnt_action = str(len(action_list))
                                if not os.path.exists(os.path.join(save_frames, cnt_action)):
                                    os.makedirs(os.path.join(save_frames, cnt_action))
                                cv2.imwrite(os.path.join(save_frames, cnt_action, '{}.jpg'.format(num)), f)
                            """
                            #print('number of sampled frames : {}'.format(len(sampled_frames)))

                            # crop all images

                            cropped_frames = np.array(CropFrames(yolo, meta, sampled_frames))

                            # zero padding in time-axis
                            maxlen = 64
                            preprocessed = np.array(cropped_frames.tolist() + [np.zeros_like(cropped_frames[0])] * (maxlen - len(cropped_frames)))

                            result, confidence, top_3 = pred_action(preprocessed)
                            print("{}, {}, {}\n".format(result, confidence, top_3))
                            ##### send to flask

                            event_end_frame = frame_num

                            action_time = time.time()  # reset action_time

                            #if True:#result != None:
                                # action_end_frame = frame_num
                                #action_list.append(result)

                            #    if len(action_list) >= args.action_thresh:
                            #        motion_detect = False
                            #        break
                            #else:
                            #    print("Do the previous action again ..\n")

                            motion_detect = False
                            sampled_frames = []
                            frames=[]

                    waiting_time = time.time() - action_time
                    if waiting_time > args.waiting_time:
                        print("waiting time : {}".format(waiting_time))
                        print("Finish detecting actions .\n")
                        break

        #cv2.imshow('frame', display_frame)
        #cv2.waitKey(50)

        #cv2.imwrite('/home/pjh/Videos/test-arbitrary_frames/{}.jpg'.format(frame_num), display_frame)

        # print(time.time() - prev_time)
        #frame_num = frame_num + 1

        # if len(frames) > args.caption_video_length:
        #     frames.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()

    # try:
    #     # requests.post('http://155.230.24.109:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
    #     requests.get(
    #         'http://192.168.0.4:3001/api/v1/actions/action/{}/{}'.format('home', action_list[-1]))
    #     # requests.post('http://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
    #     print('send action')
    # except:
    #     pass


