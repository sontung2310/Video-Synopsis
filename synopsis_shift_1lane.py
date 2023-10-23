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
parser.add_argument('-numlane', '--num_lane', default=1, type=int, help='Finish frame')
parser.add_argument('-rotate', '--rotate_angle', default=0, type=int, help='rotate angle')
parser.add_argument('-deltax', '--delta_x', default=0, type=int, help='delta_x_object')
parser.add_argument('-deltay', '--delta_y', default=0, type=int, help='delta_y_object')

args = parser.parse_args()
rotate_angle = args.rotate_angle
total_overlap_region = 0
frame_no = 0
frame_overlap = 0


def mouse_draw_pts(img):
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


def mouse_draw_polygon(img):
    polygon = []

    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            polygon.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            polygon.pop()

        if len(polygon) > 0:
            # Draw the last point in pts
            cv2.circle(img2, polygon[-1], 3, (0, 0, 255), -1)

        if len(polygon) > 1:
            #
            for i in range(len(polygon) - 1):
                cv2.circle(img2, polygon[i], 5, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
                cv2.line(img=img2, pt1=polygon[i], pt2=polygon[i + 1], color=(255, 0, 0), thickness=2)

        cv2.imshow('image', img2)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)

    print(
        "[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area")
    print("[INFO] Press ‘S’ to determine the selection area and save it")
    print("[INFO] Press ESC to quit")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return polygon


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
    cv2.imwrite("background.jpg", background)
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

        contours = sorted(contours, key=lambda contour: len(contour), reverse=True)

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
    ids = ids[5:10]  # Get 5 id in ids #taij thang choa nay`
    max_length = 0
    object_per_sec = overlap_param
    # object_per_frame = int(fps * object_per_sec * args.num_lane)  # thoi gian giua cac object
    object_per_frame = int(fps * object_per_sec)  # thoi gian giua cac object

    print("object_per_frame: ", object_per_frame)
    h, w = background.shape[:2]
    # h, w = background.size()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = args.video_path[:-4] + '_' + str(args.num_lane) + "_shift_lane.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    global frame_no
    count = 0
    while (1):
        print("frame_no: ", frame_no)
        frame = background.copy()
        global total_overlap_region
        temp = total_overlap_region
        for i in range(frame_no + 1):
            group_no = (frame_no - i) / object_per_frame
            if not group_no.is_integer(): continue
            group_no = int(group_no)
            for j, id_no in enumerate(
                    range(group_no * args.num_lane, (group_no + 1) * args.num_lane)):
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

                background_temp = np.zeros(shape=background.shape, dtype=np.uint8)
                background_temp[y:y + h, x:x + w] = object_img
                object_img = background_temp
                # cv2.imshow("asdasd",object_img)
                # cv2.waitKey()
                # (frame_no_in_ori_video, id, classes, x, y, w, h) = data_id[i]

                # line_temp = lines[j].copy()
                # line_temp = sorted(line_temp, key=lambda point: abs(x - point[0]) + abs(y - point[1]))
                # # print("xywh: ", x, y, w, h)
                # dest_point = line_temp[0]

                # x = dest_point[0]
                # y = dest_point[1]
                # delta[id][0] = delta[id][0] * 1.5

                if (x < 0): x = 0
                if (y < 0): y = 0
                if (x + w >= 1920): w = 1920 - x
                if (y + h >= 1080): h = 1080 - y
                if w <= 0 or h <= 0: continue
                # object_img = cv2.resize(object_img, (w, h))
                if j % args.num_lane == 1:
                    object_img = moveandrotate(object_img, rotate_angle=rotate_angle, delta_x=args.delta_x, delta_y=0)

                if j % args.num_lane == 0:
                    object_img = moveandrotate(object_img, rotate_angle=rotate_angle, delta_x=-args.delta_x, delta_y=0)

                gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
                _, contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0: continue
                contours = sorted(contours,
                                  key=lambda contour: cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3],
                                  reverse=True)
                x, y, w, h = cv2.boundingRect(contours[0])
                object_img = object_img[y:y + h, x:x + w]
                ratio = 0.7
                x = int(x + w * (1 - ratio) / 2)
                y = int(y + h * (1 - ratio) / 2)
                w = int(w * ratio)
                h = int(h * ratio)
                object_img = cv2.resize(object_img, (w, h))
                frame = put_object_into_frame(frame, object_img, background, x, y)
                # _,frame=image1onimage2(object_img,frame)

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


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def image1onimage2(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    mask[(img1 != [0, 0, 0]).all(-1)] = 255
    mask = 255 - mask
    img2 = cv2.bitwise_and(img2, img2, mask=mask)
    img2 = img1 + img2
    return img1, img2


def moveandrotate(img, rotate_angle, delta_x, delta_y):
    h, w = img.shape[:2]
    # delta_x = -500 * (0 + 1)
    # delta_y = 0 * 0
    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    img = cv2.warpAffine(img, M, (w, h))
    img = rotate_image(img, -rotate_angle)

    return img


def create_lane(background, num_lane, polygon_lane, rotate_angle):
    h, w = background.shape[:2]
    polygon_lane = np.array(polygon_lane)
    mask = np.zeros(shape=background.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [polygon_lane], -1, (255, 255, 255), thickness=cv2.FILLED)
    main_lane = background.copy()
    main_lane = cv2.bitwise_and(main_lane, main_lane, mask=mask)
    # shift left
    for i in range(int((num_lane - 1) / 2)):
        minor_lane = main_lane.copy()
        minor_lane = moveandrotate(minor_lane, rotate_angle=rotate_angle, delta_x=args.delta_x + 200, delta_y=0)
        minor_lane, background = image1onimage2(minor_lane, background)

    return background


def summarize():
    video = cv2.VideoCapture(args.video_path)
    if args.background_path != "":
        background = get_Background(args.background_path)
    else:
        background = get_Background(args.video_path)
    h, w = background.shape[:2]
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
        start_points = [(1553, 7),(1000,7)]  # mydinh3
    elif args.video_path == 'Hadong_ST.MOV':
        start_points = [(574, 8)]  # Hadong_ST
    elif args.video_path == 'vcc6.MOV':
        start_points = [(1267, 4)]  # vcc6.MOV

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
    data = new_data
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

    save_str = f"video path: {args.video_path}\toption: {args.num_lane}\ttotal_overlap_region: {total_overlap_region}\tframe_overlap: {frame_overlap}\tframe_no: {frame_no}\n"

    file.write(save_str)

    file.close()
