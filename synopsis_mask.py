import sys
import os

sys.path.append('centermask2')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from detectron2.engine.defaults import DefaultPredictor
from centermask.config import get_cfg
import torch
import datetime
import argparse
import cv2
import numpy as np
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Summarize Video')
parser.add_argument('-anno', '--annotation_path', type=str, help="Annotation path")
parser.add_argument('-video', '--video_path', type=str, help="Video path")
parser.add_argument('-background', '--background_path', default="", type=str, help="Background path")
parser.add_argument('-class', '--class_object', default=-1, type=int, help='Option Class')
parser.add_argument('-motion', '--motion_vector', default=-1, type=int, help='Option direction of vehicle')
parser.add_argument('-id', '--object_id', default=-1, type=int, help='Display only ID in video')
parser.add_argument('-overlap', '--overlap_param', type=float, help='Overlap')
parser.add_argument('-color', '--object_color', default=-1, type=int, help='Option color')
parser.add_argument('-start', '--frame_start', default=-1, type=int, help='Start frame')
parser.add_argument('-finish', '--frame_finish', default=-1, type=int, help='Finish frame')

args = parser.parse_args()

total_overlap_region = 0
frame_no = 0


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file('centermask2/configs/centermask/centermask_V_99_eSE_FPN_ms_3x.yaml')
    cfg.MODEL.WEIGHTS = 'centermask2/centermask2-V-99-eSE-FPN-ms-3x.pth'
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1  # faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.4
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.4
    cfg.freeze()
    return cfg


def get_MotionVector(data, id):  # 0 or 1 demo
    data_id = [x for x in data if x[1] == id]
    data_id = sorted(data_id, key=lambda x: x[0])
    # print(np.array(data_id))
    # print('\n')
    if data_id[0][3] + data_id[0][4] - data_id[-1][3] - data_id[-1][4] > 0:
        return 0
    else:
        return 1


def get_Background(video_path):  # Using median to find background
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < 200:  # Using 100 first frame to find background
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = get_Frame(cap, random.randint(0, total_frame))
        frames.append(frame)
    frames = np.array(frames, dtype=np.uint8)
    background = np.median(frames, axis=0)
    print(background.shape)
    background = np.array(background, dtype=np.uint8)
    # cv2.imshow("background", background)
    # cv2.waitKey()
    return background


def get_Anno(anno_path):  # frame,id,class,x,y,w,h,color
    data = np.loadtxt(anno_path)
    data = np.round(data)
    data = np.array(data, dtype=int)
    data = data.clip(min=0)
    return data


def get_Frame(cap, frame_no):  # Get frame in video
    cap.set(1, frame_no - 1)
    sucess, frame = cap.read()
    return frame


def display_class(data, class_object):
    class_id = [x for x in data if x[2] == class_object]
    class_id = sorted(class_id, key=lambda x: x[0])
    return np.array(class_id)


def direction_vehicle(data, motion_vector):
    motion = [x for x in data if get_MotionVector(data, x[1]) == motion_vector]
    motion = sorted(motion, key=lambda x: x[0])
    return np.array(motion)


def color_filter(data, color):
    color_id = [x for x in data if x[7] == color]
    color_id = sorted(color_id, key=lambda x: x[0])
    return np.array(color_id)


def get_object_lengthtime(data, id):  # Get object lengthtime
    return len([x for x in data if x[1] == id])


def image1OnImage2(img1, img2):  #
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    _, thresh = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    img2 = cv2.bitwise_and(img2, img2, mask=255 - thresh)
    return img1 + img2


def put_object_into_frame(frame, object_img, background, x, y, w, h):
    black_background_object = np.zeros(shape=background.shape, dtype=np.uint8)
    black_background_object[y:y + h, x:x + w] = object_img
    # cv2.imshow("black_ground",black_background_object)
    # cv2.imshow("frame",cv2.resize(frame,(800,600)))
    # cv2.imshow("single object",cv2.resize(background_temp,(800,600)))
    mask_object_onframe = np.zeros(shape=background.shape[:2], dtype=np.uint8)
    mask_object_onframe[frame[:, :, 0] != background[:, :, 0]] = 1
    # cv2.imshow("mask_object_onframe",cv2.resize(mask_object_onframe*255,(800,600)))

    # cv2.imshow("Mask Object On Frame",mask_object_onframe*255)
    mask_object = np.zeros(shape=background.shape[:2], dtype=np.uint8)
    mask_object[(black_background_object != [0, 0, 0]).all(-1)] = 1
    # cv2.imshow("mask_object",cv2.resize(mask_object*255,(800,600)))
    # cv2.waitKey()

    # cv2.imshow("Mask Object",mask_object*255)
    # overlap = np.zeros(shape=background.shape[:2], dtype=np.uint8)
    # overlap[mask_object == 255 and mask_object_onframe == 255] = 255
    overlap = cv2.bitwise_and(mask_object, mask_object_onframe) * 255
    overlap = cv2.dilate(overlap, kernel=np.ones(shape=(5, 5), dtype=np.uint8))
    # cv2.imshow("overlap",overlap)
    _, contours, _ = cv2.findContours(overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        ox, oy, ow, oh = cv2.boundingRect(contours[0])
        global total_overlap_region
        total_overlap_region += ow * oh
        overlap_region = cv2.addWeighted(frame[oy:oy + oh, ox:ox + ow], 0,
                                         black_background_object[oy:oy + oh, ox:ox + ow], 1, 0)
        frame = image1OnImage2(black_background_object, frame)
        frame[oy:oy + oh, ox:ox + ow] = overlap_region
    else:
        frame = image1OnImage2(black_background_object, frame)

    # cv2.imshow("overlap",overlap)
    # cv2.imshow("frame", frame)
    # cv2.imshow("object_img", object_img)
    # cv2.imshow("asd", mask_object)
    #
    # cv2.waitKey()
    # print("Total overlap region: ", total_overlap_region)
    return frame


# Put multi ID into the background
def multi_id(video, data, background, ids, overlap_param, centermask_model):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(fps)
    # Sort id of object from max lengthtime to min lengthtime
    ids = sorted(ids, key=lambda id: (get_object_lengthtime(data, id)), reverse=True)
    print("sorted")
    print(ids)
    # ids = [ids]
    ids = ids[28:30]  # Get 5 id in ids #taij thang choa nay`
    max_length = 0
    object_per_sec = overlap_param
    object_per_frame = int(fps * object_per_sec)  # thoi gian giua cac object
    print("object_per_frame: ", object_per_frame)
    h, w = background.shape[:2]
    # h, w = background.size()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = "demo_mask.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    global frame_no
    count = 0
    while (1):
        print("frame_no: ", frame_no)
        frame = background.copy()
        for i in range(frame_no + 1):
            id_no = (frame_no - i) / object_per_frame
            # print("id_no: ", id_no)
            if not id_no.is_integer(): continue
            id_no = int(id_no)
            if id_no >= len(ids): continue
            id = ids[id_no]
            print("id: ", id)
            fps = int(video.get(cv2.CAP_PROP_FPS))
            h, w = background.shape[:2]

            data_id = [x for x in data if x[1] == id]
            data_id = sorted(data_id, key=lambda x: x[0])
            if i >= len(data_id): continue
            # print("data_id[i]: ", data_id[i])
            # (frame_no_in_ori_video, id, classes, x, y, w, h, color) = data_id[i]
            (frame_no_in_ori_video, id, classes, x, y, w, h) = data_id[i]
            old_x, old_y, old_w, old_h = x, y, w, h
            add_ratio = 30 / 100
            w = int(w * (1 + add_ratio))
            h = int(h * (1 + add_ratio))
            x = int(x - w * add_ratio / 2)
            y = int(y - h * add_ratio / 2)
            if (x < 0): x = 0
            if (y < 0): y = 0
            if (x + w > 1920): w = 1920 - x
            if (y + h > 1080): h = 1080 - y
            # print("xywh: ", x, y, w, h)
            frame_no_img = get_Frame(video, frame_no_in_ori_video)

            object_img = frame_no_img[y:y + h, x:x + w]

            device = torch.device('cpu')
            predictions = centermask_model(object_img)
            masks = predictions['instances'].pred_masks.to(device).numpy()
            masks = np.array(masks, dtype=np.uint8) * 255

            contour = []
            if masks.shape[0] > 0:
                final_masks = np.zeros(shape=object_img.shape[:2], dtype=np.uint8)
                for mask in masks:
                    final_masks += mask
                # get biggest region
                _, contours, _ = cv2.findContours(final_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours,
                                  key=lambda contour: cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3],
                                  reverse=True)
                contour = contours[0]
                contour = cv2.convexHull(contour)
                final_masks = np.zeros_like(final_masks)
                cv2.drawContours(final_masks, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                final_masks = cv2.dilate(final_masks, np.ones((10, 10), dtype=np.uint8), iterations=1)
                object_img = cv2.bitwise_and(object_img, object_img, mask=final_masks)

                # cv2.imshow('final mask', object_img)
                # cv2.waitKey()

            frame = put_object_into_frame(frame, object_img, background, x, y, w, h)

            if contour != []:
                cv2.drawContours(frame, [contour + [x, y]], -1, (0, 255, 0), 2)
            else:

                cv2.rectangle(frame, (old_x, old_y), (old_x + old_w, old_y + old_h), (0, 255, 0), 2)

            date_time_str = '10:10:00'
            date_time_obj = datetime.datetime.strptime(date_time_str, '%H:%M:%S')
            add_time_obj = datetime.timedelta(seconds=+ int((frame_no_in_ori_video / fps)))
            date_time_obj = date_time_obj + add_time_obj
            cv2.putText(frame,
                        "ID: " + str(id) + " " +
                        "Time: {:02d}:{:02d}:{:02d}".format(date_time_obj.hour, date_time_obj.minute,
                                                            date_time_obj.second),
                        (old_x, old_y),
                        cv2.FONT_HERSHEY_COMPLEX, h / 300, (255, 0, 0), 4)

            # cv2.imwrite("objects/img_" + str(id) + "_" + str(i) + ".jpg", frame_no_img[y:y + h, x:x + w])
            # frame[y:y + h, x:x + w] = object_img

        if np.all(frame == background): count += 1
        if count > 60: break
        out.write(frame)
        frame_no += 1


def summarize():
    centermask_model = DefaultPredictor(setup_cfg())
    video = cv2.VideoCapture(args.video_path)
    if args.background_path != "":
        background = get_Background(args.background_path)
    else:
        background = get_Background(args.video_path)
    # background = cv2.resize(background,(960,540))
    print(background.shape)
    data = get_Anno(args.annotation_path)
    if args.frame_start != -1:
        data = [x for x in data if (x[0] >= args.frame_start and x[0] <= args.frame_finish)]
        data = np.array(data)
    print("original")
    print(data)
    # Class Filter
    if args.class_object != -1:
        data = display_class(data, args.class_object)
    print("class data: ", data.shape)
    # Motion Filter
    if args.motion_vector != -1:
        data = direction_vehicle(data, args.motion_vector)
    print("motion data", data.shape)
    if args.object_color != -1:
        data = color_filter(data, args.object_color)
    print("color data", data.shape)

    fps = video.get(cv2.CAP_PROP_FPS)

    object_ids = list(set(x[1] for x in data))
    object_ids = sorted(object_ids)
    # object_ids.remove(490)
    result = []
    num_objects = len(object_ids)
    # num_objects = 5
    object_ids = [object_ids[i:i + num_objects] for i in range(0, len(object_ids), num_objects)]
    print(object_ids)
    if args.object_id != -1:
        object_ids = [[args.object_id]]
    for ids in object_ids:
        multi_id(video, data, background, ids, args.overlap_param, centermask_model)  # [10]


if __name__ == '__main__':
    summarize()
    # summarize("vcc6.MOV", "annotation_vcc6.txt")

    print("Total overlap region: ", total_overlap_region)
    file = open("results_sub.txt", "a")
    file.write("Total overlap region: " + str(total_overlap_region) + " " + "Frame_no: " + str(frame_no) + "\n")

    file.close()
