import os
import argparse
import logging
import tensorflow as tf
import pathlib
import numpy as np
import torchvision.utils as vutils
from IPython.display import Image
import torch
from torchvision import transforms
from keras import backend as K
import glob
from restoration_models.mapping_model import Pix2PixHDModel_Mapping
from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results

from options.test_options import TestOptions

from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
import cv2
from u2net import U2NET

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--images', type=str, default='MP_figure_bg_rm_after')
  parser.add_argument('--model', type=str, default='pretrained/model_segmentation_skin_30.pth')

  parser.add_argument('--model-type', type=str, choices=models, default='FCNResNet101')

  parser.add_argument('--threshold', type=float, default=0.5)

  parser.add_argument('--save', action='store_true', default=True)
  parser.add_argument('--display', action='store_true', default=False)

  return parser.parse_args()

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


def enhance(img):
    sub = np.where(img > 0.2, 1, 0)
    return sub


def resize(filename, size=(224, 224)):
    im = Image.open(filename)
    im_resized = im.resize(size, Image.ANTIALIAS)
    return (im_resized)


def accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
  classification problems.
  '''
    return K.mean(K.equal(y_true, K.round(y_pred)))


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
  how many relevant items are selected.
  '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np[predict_np > 0.2] = 1
    predict_np[predict_np <= 0.2] = 0
    print(predict_np)

    # predict_np_reverse = np.where(predict_np == 1., 0., 1.)
    # print(predict_np_reverse)
    # predict_np_reverse[predict_np_reverse == 1] = 0
    # predict_np_reverse[predict_np_reverse == 0] = 1
    # exit()



    im = Image.fromarray(predict_np * 255).convert('RGB')
    # im_reverse = Image.fromarray(predict_np_reverse * 255).convert('RGB')

    image = cv2.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    # imo_reverse = im_reverse.resize((image.shape[1], image.shape[0]))

    pb_np = np.array(imo)
    perct = np.count_nonzero(pb_np == 0) / (pb_np.shape[0] * pb_np.shape[1] * pb_np.shape[2])
    # pb_np_reverse = np.array(imo_reverse)


    print(perct)
    # image_reverse = cv2.bitwise_and(image, pb_np_reverse)
    if not perct >= 0.87:


        image = cv2.bitwise_and(image, pb_np)
    #     aaa = img_name.split(".")
    #     bbb = aaa[0:-1]
    #     imidx = bbb[0]
    #
    # for i in range(1, len(bbb)):
    #     imidx = imidx + "." + bbb[i]
    # print()
    print(f'{d_dir}/{os.path.basename(image_name)}')

    cv2.imwrite(f'{d_dir}/{os.path.basename(image_name)}', image)
    # cv2.imwrite(f'{d_dir}/reversed_{os.path.basename(image_name)}', image_reverse)


def selective_mask_t(image_src, mask, channels=[]):
    mask = mask[:, torch.tensor(channels).long()]
    mask = torch.sgn(torch.sum(mask, dim=1)).to(dtype=image_src.dtype).unsqueeze(-1)

    return mask * image_src


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def main():
    opt = TestOptions().parse(save=False)
    model = Pix2PixHDModel_Mapping()
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True

    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    opt.name = "mapping_quality"
    opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
    opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")

    model.initialize(opt)
    model.eval()
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()
    for filename in os.listdir(
            '../Monkeypox lesion/low_og_img_10/Monkey_Pox'):
        print(f"##############################################################processing {filename}##############################################################################")

        f = os.path.join(
            '../Monkeypox lesion/low_og_img_10/Monkey_Pox',
            filename)
        input = Image.open(f).convert("RGB")
        input = data_transforms(input, scale=False)

        input = img_transform(input)
        input = input.unsqueeze(0)
        mask = torch.zeros_like(input)
        try:
            with torch.no_grad():
                generated = model.inference(input, mask)
        except Exception as ex:
            print(str(ex))

        image_grid = vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            f"../Monkeypox lesion/low_og_img_10_restored/Monkey_Pox/{filename}",
            nrow=1,
            padding=0,
            normalize=True,
        )
    for filename in os.listdir(
            '../Monkeypox lesion/low_og_img_10/Others'):
        print(f"##############################################################processing {filename}##############################################################################")

        f = os.path.join(
            '../Monkeypox lesion/low_og_img_10/Others',
            filename)
        input = Image.open(f).convert("RGB")
        input = data_transforms(input, scale=False)

        input = img_transform(input)
        input = input.unsqueeze(0)
        mask = torch.zeros_like(input)
        try:
            with torch.no_grad():
                generated = model.inference(input, mask)
        except Exception as ex:
            print(str(ex))

        image_grid = vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            f"../Monkeypox lesion/low_og_img_10_restored/Others/{filename}",
            nrow=1,
            padding=0,
            normalize=True,
        )

    for filename in os.listdir(
            '/Users/timothy/Desktop/Timothy-2021-2022/health_ai/Monkeypox lesion/aug_testing_dataset/Others'):
        print(f"##############################################################processing {filename}##############################################################################")
        f = os.path.join(
            '/Users/timothy/Desktop/Timothy-2021-2022/health_ai/Monkeypox lesion/aug_testing_dataset/Others',
            filename)
        input = Image.open(f).convert("RGB")
        input = data_transforms(input, scale=False)

        input = img_transform(input)
        input = input.unsqueeze(0)
        mask = torch.zeros_like(input)
        try:
            with torch.no_grad():
                generated = model.inference(input, mask)
        except Exception as ex:
            print(str(ex))

        image_grid = vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            f"restored_resolution_for_testing/Others/{filename}.jpg",
            nrow=1,
            padding=0,
            normalize=True,
        )
    # exit()
    image_dir = '/Users/timothy/Desktop/Timothy-2021-2022/health_ai/Monkeypox lesion/aug_testing_dataset/Monkey_Pox'
    image_dir = '/Users/timothy/Desktop/Timothy-2021-2022/health_ai/coco/Others'
    image_dir = 'MP_figure_og/'
    #
    # # image_dir = 'coco_restored/Others'
    # prediction_dir = 'coco_background_removal/Others'
    prediction_dir = 'MP_figure_bg_rm'
    model_dir = 'u2net_human_seg.pth'
    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(len(img_name_list))
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    model_name = 'u2net'
    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        # print(inputs_test)
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        # print(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)


        # img_mult = tf.multiply(inputs_test, pred)
        print(f'saving to {image_dir}/{img_name_list[i_test].split(os.sep)[-1]}')
        save_output(f'{image_dir}/{img_name_list[i_test].split(os.sep)[-1]}', pred, prediction_dir)

        # print(f"here {prediction_dir}")

        del d1, d2, d3, d4, d5, d6, d7

    # exit()
    args = parse_args()
    device = 'cpu'
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model', type=str, default='pretrained/model_segmentation_skin_30.pth')
    #
    # parser.add_argument('--model-type', type=str, choices=models, default='FCNResNet101')

    # parser.add_argument('--threshold', type=float, default=0.005)

    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--display', action='store_true', default=False)
    logging.info(f'loading {args.model_type} from {args.model}')
    model = torch.load(args.model, map_location=device)
    model = load_model(models[args.model_type], model)
    model.to(device).eval()

    print(f'evaluating images from {args.images}')
    image_dir = pathlib.Path(args.images)

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg']):
        # print(image_file)

        print(f'segmenting {os.path.basename(image_file)} with threshold of {args.threshold}')

        image = fn_image_transform(image_file)

        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            results = model(image)['out']
            results = torch.sigmoid(results)

            results = results > args.threshold
            print(results)

            results_reverse = ~results
            print(results_reverse)

        for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
            if args.save:
                # output_name = f'results_5/results_{category}_{image_file.name.replace(".jfif",".jpg")}'
                # print(output_name)
                # logging.info(f'writing output to {output_name}')
                # cv2.imwrite(output_name, category_image)
                print(f'saving to MP_figure_skin_seg/{os.path.basename(image_file)}')

                cv2.imwrite(f'MP_figure_skin_seg/{os.path.basename(image_file)}',
                            mask_image)  # mask_{category}_{image_file.name.replace(".jfif",".jpg")}

            if args.display:
                cv2.imshow(category, category_image)
                cv2.imshow(f'mask_{category}', mask_image)
        for category, category_image, mask_image in draw_results(image[0], results_reverse[0], categories=model.categories):
            if args.save:
                # output_name = f'results_5/results_{category}_{image_file.name.replace(".jfif",".jpg")}'
                # print(output_name)
                # logging.info(f'writing output to {output_name}')
                # cv2.imwrite(output_name, category_image)
                print(f'saving to MP_figure_skin_seg/reversed_{os.path.basename(image_file)}')

                cv2.imwrite(f'MP_figure_skin_seg/reversed_{os.path.basename(image_file)}',
                            mask_image)  # mask_{category}_{image_file.name.replace(".jfif",".jpg")}

            if args.display:
                cv2.imshow(category, category_image)
                cv2.imshow(f'mask_{category}', mask_image)
        exit()
        if args.display:
            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()
    #
    #   return parser.parse_args()

    #
    # exit()














if '__name__' == '__main__':
    main()