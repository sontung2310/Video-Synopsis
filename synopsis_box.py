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
frame_overlap = 0


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


def put_object_into_frame(frame, object_img, background, x, y, w, h):
    black_background_object = np.zeros(shape=background.shape, dtype=np.uint8)
    black_background_object[y:y + h, x:x + w] = object_img
    background_temp = background.copy()
    background_temp[y:y + h, x:x + w] = object_img
    # cv2.imshow("black_ground",black_background_object)
    # cv2.imshow("frame",cv2.resize(frame,(800,600)))
    # cv2.imshow("single object",cv2.resize(background_temp,(800,600)))
    mask_object_onframe = np.zeros(shape=background.shape[:2], dtype=np.uint8)
    mask_object_onframe[frame[:, :, 0] != background[:, :, 0]] = 1
    # cv2.imshow("mask_object_onframe",cv2.resize(mask_object_onframe*255,(800,600)))

    # cv2.imshow("Mask Object On Frame",mask_object_onframe*255)
    mask_object = np.zeros(shape=background.shape[:2], dtype=np.uint8)
    mask_object[y:y + h, x:x + w] = np.ones(shape=(h, w), dtype=np.uint8)
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
        overlap_region = cv2.addWeighted(frame[oy:oy + oh, ox:ox + ow], 0.5,
                                         black_background_object[oy:oy + oh, ox:ox + ow], 0.5, 0)
        frame[y:y + h, x:x + w] = object_img
        frame[oy:oy + oh, ox:ox + ow] = overlap_region
    else:
        frame[y:y + h, x:x + w] = object_img

    # cv2.imshow("overlap",overlap)
    # cv2.imshow("frame", frame)
    # cv2.imshow("object_img", object_img)
    # cv2.imshow("asd", mask_object)
    #
    # cv2.waitKey()
    # print("Total overlap region: ", total_overlap_region)
    return frame


# cupy
# Put multi ID into the background
def multi_id(video, data, background, ids, overlap_param):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(fps)
    # Sort id of object from max lengthtime to min lengthtime
    ids = sorted(ids, key=lambda id: (get_object_lengthtime(data, id)), reverse=True)
    print("sorted")
    print(ids)
    # ids = [ids]
    # ids = ids[10:30]  # Get 5 id in ids #taij thang choa nay`
    max_length = 0
    object_per_sec = overlap_param
    object_per_frame = int(fps * object_per_sec)  # thoi gian giua cac object
    print("object_per_frame: ", object_per_frame)
    h, w = background.shape[:2]
    # h, w = background.size()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = args.video_path[:-4] + "_reduce_size.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    global frame_no
    count = 0
    while (1):
        print("frame_no: ", frame_no)
        frame = background.copy()
        global total_overlap_region
        temp = total_overlap_region
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

            if (x < 0): x = 0
            if (y < 0): y = 0
            if (x + w > 1920): w = 1920 - x
            if (y + h > 1080): h = 1080 - y
            # print("xywh: ", x, y, w, h)

            frame_no_img = get_Frame(video, frame_no_in_ori_video)

            object_img = frame_no_img[y:y + h, x:x + w]
            ratio = 0.6
            x = int(x + w * (1 - ratio) / 2)
            y = int(y + h * (1 - ratio) / 2)
            w = int(w * ratio)
            h = int(h * ratio)
            object_img = cv2.resize(object_img, (w, h))
            frame = put_object_into_frame(frame, object_img, background, x, y, w, h)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            date_time_str = '10:10:00'
            date_time_obj = datetime.datetime.strptime(date_time_str, '%H:%M:%S')
            add_time_obj = datetime.timedelta(seconds=+ int((frame_no_in_ori_video / fps)))
            date_time_obj = date_time_obj + add_time_obj
            cv2.putText(frame,
                        "ID: " + str(id) + " " +
                        "Time: {:02d}:{:02d}:{:02d}".format(date_time_obj.hour, date_time_obj.minute,
                                                            date_time_obj.second),
                        (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, h / 300, (255, 0, 0), 4)

            # cv2.imwrite("objects/img_" + str(id) + "_" + str(i) + ".jpg", frame_no_img[y:y + h, x:x + w])
            # frame[y:y + h, x:x + w] = object_img

        global frame_overlap
        if total_overlap_region > temp:
            frame_overlap += 1

        if np.all(frame == background): count += 1
        if count > 60: break
        out.write(frame)
        frame_no += 1


def summarize():
    video = cv2.VideoCapture(args.video_path)
    if args.background_path != "":
        background = get_Background(args.background_path)
    else:
        background = get_Background(args.video_path)

    h, w = background.shape[:2]
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
    # ______________________________________________
    # start_points = mouse_draw_pts(background)

    start_points = []
    if args.video_path == 'mydinh3.mp4':
        start_points = [(1553, 7)]  # mydinh3
    elif args.video_path == 'Hadong_ST.MOV':
        start_points = [(574, 8)]  # Hadong_ST
    elif args.video_path == 'vcc6.MOV':
        start_points = [(1267, 4)]  # vcc6

    new_data = []
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = "1234.mp4"
    output_anno_path = '1234.txt'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    ids = list(set(x[1] for x in data))
    for id in ids:
        data_id = [x for x in data if x[1] == id]
        data_id = sorted(data_id, key=lambda x: x[0])
        (frame_no_in_ori_video, id, classes, x, y, w, h) = data_id[0]
        if x < start_points[0][0] and start_points[0][0] < x + w:
            for i in range(len(data_id)):
                new_data.append(data_id[i])

    new_data = np.array(new_data)
    # data = new_data
    # _________________________________________________
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
        multi_id(video, data, background, ids, args.overlap_param)  # [10]


if __name__ == '__main__':
    summarize()
    # summarize("vcc6.MOV", "annotation_vcc6.txt")

    print("Total overlap region: ", total_overlap_region)
    file = open("results_sub.txt", "a")

    save_str = f"video path: {args.video_path}\toption: {'reduce_size'}\ttotal_overlap_region: {total_overlap_region}\tframe_overlap: {frame_overlap}\tframe_no: {frame_no}\n"

    file.write(save_str)

    file.close()
