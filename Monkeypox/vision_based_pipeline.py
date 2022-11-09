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


def parse_args(bg_rm_dir_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default=bg_rm_dir_name)
    parser.add_argument('--model', type=str, default='skin_seg_pretrained/model_segmentation_skin_30.pth')

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


def save_output(read_image_name, pred, save_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np[predict_np > 0.2] = 1
    predict_np[predict_np <= 0.2] = 0
    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = cv2.imread(read_image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    perct = np.count_nonzero(pb_np == 0) / (pb_np.shape[0] * pb_np.shape[1] * pb_np.shape[2])
    if not perct >= 0.87:
        image = cv2.bitwise_and(image, pb_np)

    cv2.imwrite(f'{save_dir}/{os.path.basename(read_image_name)}', image)


def selective_mask_t(image_src, mask, channels=[]):
    mask = mask[:, torch.tensor(channels).long()]
    mask = torch.sgn(torch.sum(mask, dim=1)).to(dtype=image_src.dtype).unsqueeze(-1)

    return mask * image_src


def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def main():
    opt = TestOptions().parse(save=False)
    model = Pix2PixHDModel_Mapping()
    opt.serial_batches = True
    opt.no_flip = True
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

    starting_dir_name = 'sample'
    restore_dir_name = 'sample_restored'
    bg_rm_dir_name = 'sample_background_remove'
    skin_seg_dir_name = 'sample_segmented_skin'
    print(f'Resolution Restoration @ {starting_dir_name}')
    for filename in os.listdir(
            starting_dir_name):
        print(f"##############################################################restoring {filename}##############################################################################")
        input = Image.open(os.path.join(starting_dir_name,filename)).convert("RGB")
        input = data_transforms(input, scale=False)
        input = img_transform(input)
        input = input.unsqueeze(0)
        mask = torch.zeros_like(input)
        try:
            with torch.no_grad():
                generated = model.inference(input, mask)
        except Exception as ex:
            print(str(ex))
        print(f'##############################################################saving to {restore_dir_name}/{filename}##############################################################')
        image_grid = vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            f"{restore_dir_name}/{filename}",
            nrow=1,
            padding=0,
            normalize=True,
        )

    model_dir = 'u2net_human_seg.pth'
    img_name_list = glob.glob(restore_dir_name + os.sep + '*')
    print(f'Background Removal @ {img_name_list}')
    test_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_dataloader = DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    net = U2NET(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    for i_test, data_test in enumerate(test_dataloader):

        print(f"##############################################################removing bg for {img_name_list[i_test].split(os.sep)[-1]}##############################################################################")
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]
        pred = normPRED(pred)


        print(f'##############################################################saving to {bg_rm_dir_name}/{img_name_list[i_test].split(os.sep)[-1]}##############################################################')
        save_output(f'{restore_dir_name}/{img_name_list[i_test].split(os.sep)[-1]}', pred, bg_rm_dir_name)
        del d1, d2, d3, d4, d5, d6, d7

    args = parse_args(bg_rm_dir_name)
    device = 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--display', action='store_true', default=False)
    logging.info(f'loading {args.model_type} from {args.model}')
    model = torch.load(args.model, map_location=device)
    model = load_model(models[args.model_type], model)
    model.to(device).eval()

    print(f'Skin Segmentation @ {args.images}')
    image_dir = pathlib.Path(args.images)

    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for image_file in find_files(image_dir, ['.png', '.jpg', '.jpeg', 'jfif']):
        print(f'##############################################################segmenting {os.path.basename(image_file)} with threshold of {args.threshold}##############################################################')
        image = fn_image_transform(image_file)
        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            results = model(image)['out']
            results = torch.sigmoid(results)
            results = results > args.threshold
            # results_reverse = ~results
            # print(results_reverse)
        for category, category_image, mask_image in draw_results(image[0], results[0], categories=model.categories):
            if args.save:
                print(f'##############################################################saving to {skin_seg_dir_name}/{os.path.basename(image_file)}##############################################################')
                cv2.imwrite(f'{skin_seg_dir_name}/{os.path.basename(image_file)}',mask_image)

if __name__ == '__main__':
    main()
