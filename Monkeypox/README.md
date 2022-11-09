# AICOM-MP

![](/Users/timothy/Desktop/Timothy-2021-2022/health_ai/AICOM/Monkeypox/images/MP_screening_pipeline.png)
## Datasets Description
* **_AICOM_MP_dataset_not_balanced_**: Images from _**Augmented Images**_  and our Monkeypox/healthy images (augmented)
  * NOT balanced
* **_balanced_df_train_**: images generated for balancing training dataset
  * `initial_balance_train()` function in `model.py`
* **_balanced_df_val_test_**: images generated for balancing testing and validation dataset
  * `initial_balance_df()` function in `model.py`
* _AICOM_MP_dataset_not_balanced_ + _balanced_df_train_ + _balanced_df_val_test_ = Final **_AICOM_MP_dataset_**
* _**Augmented Images**_, _**Original Images**_: datasets downloaded from [here](https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset)
* **_AICOM_LowRes1/2_**, _**AICOM_LowRes1/2_Restored**_: datasets for the restoration experinments
* _**sample(\_\*)**_: demo-datasets for demonstrating how our pipeline processes images
* _**COCO_MP**_: experimental dataset for our pipeline ablation study
* **_new_monkeypox, new_normal, normal_face, normal_hands_**: newly found images from various sources
* _**6124_testing_dataset**_: dataset used for evaluate the existing monkeypox models' performance.
## Code Description
* `model.py`
  * `main()`: model training
  * `aug_new_found_img()`: augmenting methods (13 folds) applied on newly found images (monkeypox images, healthy pictures, etc)
  * `create_testing_dataset()` or `model_create_testing_images.py`: augmenting method used to generate _**6124_testing_dataset**_
* `model_testing.py`: evalute model's performance (define in `model`) on dataset of your choise (define in `testing_dir`)
* `model_create_testing_images.py`: augmenting methods used to generate _**6124_testing_dataset**_, same function as `create_testing_dataset()`
* `vision_based_pipeline.py`: analytical pipeline for image processing/Monkeypox screening

## AICOM_MP Model Usage
* Download the weights from [Link_to_download_AICOM_MP_model_weight](#Link_to_download_AICOM_MP_model_weight)
* `python model.py`

## Pipeline Usage
* Download the necessary weights folders from [Relevant_Github_Links](#Relevant_Github_Links) to the following position
  * _AICOM_
    * _Monkeypox_
      * _skin_seg_pretrained_
      * _u2net_human_seg.pth_
      * _checkpoints_
* `python vision_based_pipeline.py`
## Relevant_Github_Links
* [Background Removal](https://github.com/xuebinqin/U-2-Net)
  * [Link](https://mcgill-my.sharepoint.com/:u:/g/personal/tianyi_yang_mail_mcgill_ca/EYQRUfCIf4RGi2YoxP8xXo8BpsJxZx_OML66KIgoWbk-2A?e=vzhXvK) to download U2 net weight.
* [Skin Segmentation](https://github.com/WillBrennan/SemanticSegmentation)
  * [Link](https://mcgill-my.sharepoint.com/:f:/g/personal/tianyi_yang_mail_mcgill_ca/EiHoyOGpCttOlkFGnEchZ80B7S6lGrTMswjL60HqU1ziew?e=hdEVuj) to download skin segmentation weight
* [Resolution Restoration](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/tree/master/Global)
  * [Link](https://mcgill-my.sharepoint.com/:f:/g/personal/tianyi_yang_mail_mcgill_ca/EhK_pyH0tbtIl-2txR0fgm8B6BLuZM2Gphvt_g-r74tePg?e=vVhtRo) to download Resolution Restoration weight
## Link_to_download_AICOM_MP_model_weight
* [AICOM_MP](https://mcgill-my.sharepoint.com/:u:/g/personal/tianyi_yang_mail_mcgill_ca/EXjzi4AU1ytKrsvaXsZf8jMBAtKZLOpZbzv4jorjOF3BCg?e=phPJkJ)
