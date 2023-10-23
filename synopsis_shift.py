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


def getEquidistantPoints(p1, p2, parts):
    return list(zip(np.linspace(p1[0], p2[0], num=parts, dtype=int),
               np.linspace(p1[1], p2[1], num=parts, dtype=int)))


def mouse_draw(img):
    pts = []

    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()

        cv2.imshow('image', img2)

        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # Left click, select point
            pts.pop()

        for point in pts:
            cv2.circle(img2, point, 3, (0, 0, 255), -1)

        cv2.imshow('image', img2)

    # Create images and windows and bind windows to callback functions
    img = img
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)

    print(
        "[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area")
    print("[INFO] Press ‘S’ to determine the selection area and save it")
    print("[INFO] Press ESC to quit")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(pts)
    return pts  # [(x,y)]


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
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    start = 0
    finish = total_frame
    if args.frame_start:
        start = args.frame_start
    if args.frame_finish:
        finish = args.frame_finish

    while cap.isOpened() and len(frames) < 200:  # Using 100 first frame to find background
        frame = get_Frame(cap, random.randint(start, finish))
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


def put_object_into_frame(frame, object_img, background, x, y):
    h, w = object_img.shape[:2]
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


def pre_process(data, ids, center_points):
    result_data = data.copy()
    K = center_points.shape[0]
    for i, id in enumerate(ids):
        i = i % K
        data_id = [x for x in result_data if x[1] == id]
        data_id = sorted(data_id, key=lambda x: x[0])
        p1 = center_points[i][0]
        p2 = center_points[i][1]
        lines = getEquidistantPoints(p1, p2, len(data_id))
        for j in range(len(data_id)):
            data_id[j][3] = lines[j][0]
            data_id[j][4] = lines[j][1]
            # result_data.append(data_id[j])
    return result_data


# Put multi ID into the background
def multi_id(video, data, background, ids, overlap_param, center_points):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(fps)
    # Sort id of object from max lengthtime to min lengthtime
    ids = sorted(ids, key=lambda id: (get_object_lengthtime(data, id)), reverse=True)
    print("sorted")
    print(ids)
    # ids = [ids]
    ids = ids[5:10]  # Get 5 id in ids #taij thang choa nay`
    max_length = 0
    object_per_sec = overlap_param
    object_per_frame = int(fps * object_per_sec)  # thoi gian giua cac object
    print("object_per_frame: ", object_per_frame)
    h, w = background.shape[:2]
    # h, w = background.size()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = "demo_1.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    global frame_no
    count = 0

    data_xy = pre_process(data, ids, center_points)

    while (1):
        print("frame_no: ", frame_no)
        frame = background.copy()
        for i in range(frame_no + 1):
            group_no = (frame_no - i) / object_per_frame
            if not group_no.is_integer(): continue
            group_no = int(group_no)
            for j, id_no in enumerate(range(group_no * center_points.shape[0], (group_no + 1) * center_points.shape[0])):
                if id_no >= len(ids): continue
                id = ids[id_no]

                print("center_no and id: ", j, id)
                fps = int(video.get(cv2.CAP_PROP_FPS))
                # h, w = background.shape[:2]

                data_id = [x for x in data if x[1] == id]
                data_id = sorted(data_id, key=lambda x: x[0])
                if i >= len(data_id): continue
                (frame_no_in_ori_video, id, classes, x, y, w, h) = data_id[i]
                frame_no_img = get_Frame(video, frame_no_in_ori_video)
                object_img = frame_no_img[y:y + h, x:x + w]
                # cv2.imshow("asdasd",object_img)
                # cv2.waitKey()
                (frame_no_in_ori_video, id, classes, x, y, w, h) = data_id[i]

                line_temp = lines[j].copy()
                line_temp = sorted(line_temp, key=lambda point: abs(x - point[0]) + abs(y - point[1]))
                # print("xywh: ", x, y, w, h)
                dest_point = line_temp[0]

                x = dest_point[0]
                y = dest_point[1]
                delta[id][0] = delta[id][0] * 1.5

                data_id = [x for x in data_xy if x[1] == id]
                data_id = sorted(data_id, key=lambda x: x[0])
                (_, _, _, x, y, _, _) = data_id[i]

                if (x < 0): x = 0
                if (y < 0): y = 0
                if (x + w >= 1920): w = 1920 - x
                if (y + h >= 1080): h = 1080 - y
                if w <= 0 or h <= 0: continue
                object_img = cv2.resize(object_img, (w, h))

                frame = put_object_into_frame(frame, object_img, background, x, y)

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

        if np.all(frame == background): count += 1
        if count > 60: break
        out.write(frame)
        frame_no += 1


# def seperate_lane(data, object_ids, center_points):
#     """
#
#     :param data: anno
#     :param center_points: [(x,y),]
#
#     :return: data for each lane and delta_x,delta_y for shifting
#     """
#     K = len(center_points)
#     result_data = [[] for i in range(K)]
#     for id in object_ids:
#
#     return result_data

def summarize():
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

    object_ids = list(set(x[1] for x in data))
    object_ids = sorted(object_ids)

    # center_points = mouse_draw(background)
    center_points=[(490, 4), (1698, 985), (229, 5), (1169, 989)]

    center_points = np.reshape(np.array(center_points), newshape=(int(len(center_points) / 2), 2, 2))
    print(center_points)
    # lines=[]
    # for two_point in center_points:
    #     lines.append(list(getEquidistantPoints(two_point[0],two_point[1])))
    # print(lines)
    # background_test=background.copy()
    # for line in lines:
    #     for point in line:
    #         cv2.circle(background_test,point,5,(0,0,255),4)
    # cv2.imshow("asd",background_test)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # center_points = [(681, 5), (842, 9), (1007, 4), (1178, 7)]
    # for point in center_points:
    #     cv2.circle(background, point, 5, (0, 0, 255), 3)
    # data = seperate_lane(data, object_ids, center_points)
    # print("length data: ",len(data))
    # for i in range(len(center_points)):
    #     print('length of lane {lane} {len}'.format(lane=str(i),len=len(data[i])))
    #
    # cv2.waitKey()

    fps = video.get(cv2.CAP_PROP_FPS)

    # object_ids.remove(490)
    result = []
    num_objects = len(object_ids)
    # num_objects = 5
    object_ids = [object_ids[i:i + num_objects] for i in range(0, len(object_ids), num_objects)]
    print(object_ids)
    if args.object_id != -1:
        object_ids = [[args.object_id]]
    for ids in object_ids:
        multi_id(video, data, background, ids, args.overlap_param, center_points)  # [10]


if __name__ == '__main__':
    summarize()
    # summarize("vcc6.MOV", "annotation_vcc6.txt")

    print("Total overlap region: ", total_overlap_region)
    file = open("results_sub.txt", "a")
    file.write("Total overlap region: " + str(total_overlap_region) + " " + "Frame_no: " + str(frame_no) + "\n")

    file.close()
