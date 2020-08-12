import argparse
import copy
import os
import sys

import align.detect_face
import numpy as np
from scipy import misc
import tensorflow as tf

import facenet


def main(args):
    dection()


def dection():
    # 将目标图片文件夹下的图片地址append进list,传入load_and_align_data(),对图片进行切割（因为其图片参数为list）
    # 这里的位置改为test_img文件夹的绝对路径
    img_dir = './train_dir/test_img/'
    img_path_set = []
    for file in os.listdir(img_dir):
        single_img = os.path.join(img_dir, file)
        print(single_img)
        print('loading...... :', file)
        img_path_set.append(single_img)

    images = load_and_align_data(img_path_set, 160, 22, 1.0)

    # 改为emb_img文件夹的绝对路径
    emb_dir = './train_dir/emb_img/'

    if (os.path.exists(emb_dir) == False):
        os.mkdir(emb_dir)

    count = 0
    for file in os.listdir(img_dir):
        misc.imsave(os.path.join(emb_dir, file), images[count])
        count = count + 1


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        with session.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(session, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        print(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        # img = misc.imread(image, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        w = abs(bb[0] - bb[2])
        h = abs(bb[1] - bb[3])
        if w > h:
            D = abs(w - h)
            newx1 = int(bb[0])
            newx2 = int(bb[2])
            newy1 = int(bb[1] - D / 2)
            newy2 = int(bb[3] + D / 2)
            if newy1 < 0:
                newy1 = 0
            if newy2 > img.shape[0]:
                newy2 = img.shape[0]
                # img.shape[0]：图像的垂直尺寸（高度）
                # img.shape[1]：图像的水平尺寸（宽度）
                # img.shape[2]：图像的通道数
        else:
            D = abs(w - h)
            newx1 = int(bb[0] - D / 2)
            newx2 = int(bb[2] + D / 2)
            newy1 = int(det[1])
            newy2 = int(det[3])
            if newx1 < 0:
                newx1 = 0
            if newx2 > img.shape[1]:
                newx2 = img.shape[1]
                # img.shape[0]：图像的垂直尺寸（高度）
                # img.shape[1]：图像的水平尺寸（宽度）
                # img.shape[2]：图像的通道数
        print(newy1,newy2, newx1,newx2)
        cropped = img[newy1:newy2, newx1:newx2, :]
        # cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

        # 根据cropped位置对原图resize，并对新得的aligned进行白化预处理
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=160)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
