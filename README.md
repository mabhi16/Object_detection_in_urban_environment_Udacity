# Object Detection in an Urban Environment

## Project overview

The main problem statement of this project is to achieve 2D object detection in an urban environment. Objects under consideration for detection are vehicles, pedestrians, and cyclists. Object detection plays an important part in the self-driving car system as it mainly helps in collision avoidance and maneuver planning.

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```
## Setup

Tf Object Detection API relies on config files. The config that we will use for this project is pipeline.config, which is the config for a SSD Resnet 50 640x640 model. 
Initialy download the pretrained model and move it to /home/workspace/experiments/pretrained_model/. 

which can be achieved by by given instruction below:
```
cd /home/workspace/experiments/pretrained_model/

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```
later, we need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
cd /home/workspace/

python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file called pipeline_new.config will be created in the /home/workspace/ directory. Move this file to the /home/workspace/experiments/reference/ directory using 
```
mv pipeline_new.config /home/workspace/experiements/reference/
```
Now we can start training process by executing below instruction
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the Training is done, evaluation process can be implemeted using below instruction 
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```
Finally the metrics and results of training and evaluation can be viewed by launching tensorboard as below
```
python -m tensorboard.main --logdir experiments/reference/
```
## Dataset Analysis 

Dataset analysis is done in order to observe the nature of the dataset, so that such an observation can be included while training the network. The data analysis in this project is done in the program 'Exploratory Data Analysis.ipynb' .In this program training image data is loaded and then randomly shuffled and out of it 10 images are selected and finally displayed with ground truth bounding boxes, which are color coded as per the class. By viewing the images one can confirm that the input pipeline for the network is functioning properly and also can observe the different features of the image like lighting, saturation, contrast, occlutions etc,. Below is one such output obtained from the program.

![img](https://user-images.githubusercontent.com/49077871/191368900-c5a64831-66f2-463d-9be5-c95bc053716e.png)

## Cross Validation

Cross-validation is a statistical method used to estimate the performance (or accuracy) of machine and deep learning models. It is used to protect against overfitting in a predictive model, particularly in a case where the amount of data may be limited. In order to evaluate our model on the go while training, at the end of the each epoch a validation task is executed. For this purspose the dataset is split into three parts train, test and val. Val contains data that are seen by the network during the training so with evaluating the network wiuth such data one can infer how good the network is regularized. Th popular data split strategy is 7:2:1 or 8:1:1. In my  case I have used 7:2:1 because during validation a higher vareity of data ensures better regulariztion.

## Training 

Reference Experiment: Below tensorboard indicate the results of the reference experiment, config file - pipeline_ref.config can be found in experiments/reference/ folder.

![metric_loss_ref](https://user-images.githubusercontent.com/49077871/191373972-5d9e9eb1-7bcd-4583-81dd-97cbb782a411.png)

Training metric for the reference experiment can be found in the text file metric.txt

It can be observed from the graph the loss is very high, this is because initial learning rate of the mometum optimizer (0.04) is quite high so when the network starts learning gradient descent happens in the larger steps and due to which the instead of converging to global minima the descent starts to oscillate between the either side of the gradient curve hence the network becomes incapable of learning anything from training and same thing is refelected in the loss.

Improved experiment: Below tensorboard indicate the results of the improvement from reference experiment, config file - pipeline_new.config can be found in experiments/reference/ folder.

![metric_loss_aug](https://user-images.githubusercontent.com/49077871/191378114-c16dce4b-e223-4053-91fd-b757a7518650.png)

Training metric for the improvement from reference experiment can be found in the text file metric.txt

To improve the training process initially, I changed the from momentum optimizer to adam optimizer and reduced initial learning rate to 0.0001, along with the cosine decay for every 200 steps provides a sufficient level of learning rate for gradient descent to smoothly reach close to global minima hence network learns to generate appropriate anchor boxes for the given feature, since I observed that there were different lighting condition in the dataset I also have included few additional augmentations namely : random_rgb_to_gray, random_distort_color, random_adjust_brightness,  random_adjust_contrast, random_adjust_hue, random_adjust_saturation. These augmentations additionally help in further reduction of training and validation loss, which is reflected in the metrics. below there also a inference video of the trained network. 

https://user-images.githubusercontent.com/49077871/191381199-8a6f2b00-d6b1-4e42-83bf-eb3a838309c1.mp4



