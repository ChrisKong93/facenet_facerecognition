import os

import cv2
from mtcnn.mtcnn import MTCNN


def find_face_detection(single_img='train_dir/test_img/Kong_Weiye.jpg'):
    img = cv2.imread(single_img)
    detector = MTCNN()
    face = detector.detect_faces(img)
    print(face)
    face = face[0]
    # 画框
    box = face["box"]
    # box = [130, 31, 120, 110]
    # box：[x, y, width, height]，x，y是人脸框左上角坐标的位置，width是框的宽度，height是框的高度。
    if box[2] > box[3]:
        length_difference = box[2] - box[3]
        # I = cv2.rectangle(img, (box[0], int(box[1] - (length_difference / 2))), (box[0] + box[2], int(box[1] - (length_difference / 2)) + box[2]), (255, 0, 0), 2)
        cropped = img[int(box[1] - (length_difference / 2)):(int(box[1] - (length_difference / 2)) + box[2]),
                  box[0]:(box[0] + box[2])]
    else:
        length_difference = box[3] - box[2]
        # print(length_difference)
        # I = cv2.rectangle(img, (int(box[0] - (length_difference / 2)), box[1]), (int(box[0] - (length_difference / 2)) + box[3], box[1] + box[3]), (255, 0, 0), 2)
        cropped = img[box[1]:(box[1] + box[3]),
                  (int(box[0] - (length_difference / 2))):(int(box[0] - (length_difference / 2)) + box[3])]
    # I = cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
    # print((int(box[0] - length_difference / 2), box[1]), (box[0] + box[3], box[1] + box[3]))
    # 画关键点
    # left_eye = face["keypoints"]["left_eye"]
    # right_eye = face["keypoints"]["right_eye"]
    # nose = face["keypoints"]["nose"]
    # mouth_left = face["keypoints"]["mouth_left"]
    # mouth_right = face["keypoints"]["mouth_right"]
    #
    # points_list = [(left_eye[0], left_eye[1]),
    #                (right_eye[0], right_eye[1]),
    #                (nose[0], nose[1]),
    #                (mouth_left[0], mouth_left[1]),
    #                (mouth_right[0], mouth_right[1])]
    # for point in points_list:
    #     cv2.circle(I, point, 1, (255, 0, 0), 4)
    # 保存

    # cv2.imwrite('result.jpg', cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return cropped


def dection():
    # 将目标图片文件夹下的图片地址append进list,传入load_and_align_data(),对图片进行切割（因为其图片参数为list）
    # 这里的位置改为test_img文件夹的绝对路径
    img_dir = './train_dir/test_img/'
    img_path_set = []
    # 改为emb_img文件夹的绝对路径
    emb_dir = './train_dir/emb_img/'
    if (os.path.exists(emb_dir) == False):
        os.mkdir(emb_dir)
    for file in os.listdir(img_dir):
        single_img = os.path.join(img_dir, file)
        print(single_img)
        print('loading...... :', file)
        cropped = find_face_detection(single_img)
        cropped = cv2.resize(cropped, (160, 160))
        cv2.imwrite(emb_dir + file, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        # img_path_set.append(single_img)
    # images = load_and_align_data(img_path_set, 160, 44, 1.0)


if __name__ == '__main__':
    dection()
