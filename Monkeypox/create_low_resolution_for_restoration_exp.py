def create_low_res(img_dir):
    classlist_train = [f for f in os.listdir(img_dir) if not f.startswith('.')]
    if not os.path.isdir('low_og_img_10'):
        os.mkdir('low_og_img_10')
    if not os.path.isdir('low_og_img_10/Others'):
        os.mkdir('low_og_img_10/Others')
    if not os.path.isdir('low_og_img_10/Monkey_Pox'):
        os.mkdir('low_og_img_10/Monkey_Pox')
    print(classlist_train)
    for klass in classlist_train:
        if klass == 'Others':
            classpath = os.path.join(img_dir, klass)
            save_classpath = os.path.join('low_og_img_10', klass)
            flist = os.listdir(classpath)
            for f in flist:
                print(f)

                abs_f_path = os.path.join(classpath,f)

                print(f"*****************Reading images from {os.path.join(classpath,f)}*****************")
                # exit()
                img_quality = 28
                while img_quality >= 10:
                    print(f'current image quality: {img_quality}')
                    img = Image.open(abs_f_path).convert('RGB')
                    save_f_path = os.path.join(save_classpath, f"quality_{img_quality}_{f}")
                    print(f'------------------saving to {save_f_path}------------------')
                    img.save(f'{save_f_path}', quality=img_quality)
                    img_quality -= 2
def main():
    create_low_res('AICOM_LowRes1')
    exit()


if __name__ == '__main__':
    main()