# Lifelong Adaptive Machine Learning for Sensor-based Human Activity Recognition Using Prototypical Networks

This is the research repository for Lifelong Adaptive Machine Learning for Sensor-based Human Activity Recognition Using Prototypical Networks. It contains the source code for *LAPNet-HAR* framework and all the experiments to reproduce the results in the paper.

## System Requirements

The deep learning system is written in `python 3`, specifically `pytorch`.

## Datasets

*LAPNet-HAR* is evaluated on 5 widely used publicly available HAR datasets:

|**Datasets** | **Activity Type** | **# of Sensor Channels** | **# of Classes** | **Balanced**|
|-------------|-------------------|--------------------------|------------------|-------------|
|[Opportunity]() | Daily Gestures | 113 | 17 | &#x2715;|
|[PAMAP2]() | Physical Activities | 52 | 12 | &#2715;|
|[DSADS]() | Daily & Sports Activities | 45 | 19 | &#2713;|
|[Skoda]() | Car Maintenance Gestures | 30 | 10 | &#2715;|
|[HAPT]() | Daily Activities & Postural Transitions | 6 | 12 | &#9746;|

## Pretrained Models

The Convolutional Neural Network pretrained on AudioSet that we use as a feature extractor can be downloaded [here](https://zenodo.org/record/3576403#.XveBmZM2rOQ). The one used in the paper is `Cnn14_mAP=0.431.pth`. 

## Scripts 

### List of scripts:

- [data.py](data.py): includes the needed classes for loading the data while keeping track of participants, sessions, and activities. 
- [utils.py](utils.py): includes helper functions.
- [models.py](models.py): includes the neural networks, implemented using `pytorch`. The model used in our paper is `FineTuneCNN14`, though the script includes other models we experimented with.
- [inference.py](inference.py): loads saved models and runs inference, originally written for Leave-One-Participant-Out evaluation.
- [location_context_inference.py](location_context_inference.py): implements the location context inference analysis (Section 8.4 in the paper), i.e. inferring the location of the device from the predicted activities. 
- [voice_band_filtering.py](voice_band_filtering.py): implements voice interaction masking using REPET method (Section 8.3 in the paper) and saves the filtered data. It is an interactive script that asks to determine the lower and upper time range for where to apply the masking. 
- [main_LOPO.py](main_LOPO.py), [main_LOSO.py](main_LOSO.py), [main_personalized_LOPO+1.py](main_personalized_LOPO+1.py): main scripts that run training as well as inference after training for Leave-One-Participant-Out (LOPO), Leave-One-Session-Out (LOSO), and LOPO + 1 session personalized analyses respectively. To run the scripts with required arguments, check the next [section](#running-the-main-scripts).

### Running the Scripts:

**Note that all following scripts run location-free modelling i.e. we assume the location of the device is unknown and thus train the model on all 19 classes. To switch to location-specific modelling and to speciy the location (kitchen, living room, or bathroom) add `--context_location='kitchen'` to any of the commands below.**

#### Leave-One-Participant-Out
To run LOPO training and evaluation using the downloaded dataset, you can run `sudo bash runme_LOPO.sh` or more specifically, determine the data type you're using by setting DATATYPE with one of the folder names in the dataset. 

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOPO.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'
```
This will save LOPO models for later user as well as the inference results saved in a CSV file. The training and testing losses as well as a confusion matrix will also be plotted at the end. 
To use the mid-interaction segments, change DATATYPE to `mid-interaction_segments` to point to the corresponding data folder.

#### Personalized Analysis

This analysis relates to Section 7.3 in the paper. We run two types of analysis: (1) Leave-One-Session-Out (LOSO) analysis which essentially trains on one session and tests on the other for every participant, ultimately creating personalized models per participant, and (2) LOPO + 1 session which for every target user, the training data consisted of data from all other users in addition to data from one session of the target user, while the test data consisted of data from the other session.

To run both, you can run `sudo bash runme_personalized_analysis.sh` or more specifically:

To run LOSO:
```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOSO.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --pad='wrap'
``` 

To run LOPO + 1 session:
```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_personalized_LOPO+1.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --pad='wrap'
``` 
#### Voice Interaction Masking

This analysis relates to Section 8.3 in the paper. The filtered data is included in the downloaded dataset under `voice_interaction_masked` folder. If you would like to apply the REPET voice masking on your own data, you can run the following script [voice_band_filtering.py](voice_band_filtering.py). The script is interactive in that for every audio wav file, you will be asked to input the time range (lower and upper bound) over which to apply the masking. Simply run `python3 voice_band_filtering.py`.

To run training using any of the above methods, simply set `DATATYPE="voice_interaction_masked"`.

#### Location Context Inference

This analysis relates to Section 8.4 in the paper. Using the LOPO models saved after training, you can run `sudo bash runme_location_inference.sh` or more specifically:
```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 location_context_inference.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='RaspiCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --model_path='FineTuneCNN14' --num_samples=4 --pad='wrap'
```

Note that the argument `--num_samples=4` determines the number of randomly selected audio clips from the hold-out participant for each context (kitchen, living room,and bathroom).

#### Recognition Performance vs. Audio Length

This analysis relates to Section 8.2 in the paper. Although you can modify the original scripts above, to easily run training and evaluation using varying audio length, you can run the following commands:

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOPO_VaryingAudioLength.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'
```

Although not included in the paper, you can run a similar analysis using LOSO:

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOSO_VaryingAudioLength.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'
```

#### Inference

Although the main scripts include inference at the end, if you want to only run inference using the saved models, you can run `sudo bash runme_inference.sh` or more specifically:

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 -i inference.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='RaspiCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --model_path='FineTuneCNN14' --num_samples=4 --pad='wrap'
```
## Audio Capture Device (Raspberry Pi)

To capture voice-based interactions with the Google Home, we developed an audio recording add-on accessory device that does not interfere with the assistant’s operation and functionalities. We included the hardware and software components needed for anyone who wants to recreate the device.

### Hardware

![Hardware setup of the audio capture device](/raspi-scripts/images/equipment.png)

1. [Raspberry Pi 3 B+](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/)
2. [ReSpeaker Linear 4 Mic Array](https://respeaker.io/linear_4_mic_array/)
3. [Raspberry Pi Camera Module v2](https://www.raspberrypi.org/products/camera-module-v2/)
4. [16GB Micro SD Card (Class 10)](https://www.amazon.com/SanDisk-COMINU024966-16GB-microSD-Card/dp/B004KSMXVM)
5. [5V 2.5A Switching Power Supply (Highly Recommended)](https://www.adafruit.com/product/1995)
6. [Google Home Mini](https://store.google.com/us/product/google_home_mini_first_gen)
7. [3D Printed Mounting Structure](/raspi-scripts/GCode_3D_Structures/): if you have access to a 3D printer and would like to create the mounting structure, we have provided the GCode files.  

### Software and Configuration

![Logic Flow Implementation](/raspi-scripts/images/logic_flow.png)

#### Install Raspbian OS

There's an entire process that documents [how to flash your SD card with the latest Raspbian OS](https://www.raspberrypi.org/documentation/installation/installing-images/).

#### Install Microphone Drivers

Next, install the audio drivers for the ReSpeaker Linear 4-Mic Array. Detailed documentation about the process [can be found here](https://wiki.seeedstudio.com/ReSpeaker_4-Mic_Linear_Array_Kit_for_Raspberry_Pi/).

```bash
$ sudo apt-get update
$ sudo apt-get upgrade
$ git clone https://github.com/respeaker/seeed-voicecard.git
$ cd seeed-voicecard
$ sudo ./install.sh  
$ sudo reboot
```
#### Connect Camera Module

Next, connecting the camera module is straightforward and you can follow this [tutorial](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera).

#### Install Requirements

These installations are required to be able to run the main software script that runs the camera and microphone as well as the mailing script.  

```bash
$ sudo pip3 install numpy 
$ sudo pip3 install pyaudio
$ sudo pip3 install picamera
$ sudo pip3 install APScheduler
$ sudo pip3 install psutil
$ sudo pip3 install email-to
```

#### Software Scripts

Make sure to upload all scripts in [raspi-scripts](/raspi-scripts) to your raspberry pi under `/home/pi/`. 

##### Mailing Script

In order to monitor the system for unexpected hardware issues, the device is set up to report the hardware activity status and process logs every 10 minutes via email. Make sure you set the variables `to`, `gmail_user`, and `gmail_password` in [mailytime.py](/raspi-scripts/mailytime.py) to the gmail where you would like to receive the device status. In order to work, you would need to turn on the "Less secure app access" on your account ([for more  info](https://support.google.com/accounts/answer/6010255#zippy=%2Cif-less-secure-app-access-is-on-for-your-account)). We recommend you create a new account to be used only for this purpose. 

##### Run ALL Autonomously On Boot

In order to provide a plug-and-play functionality, we programmed the raspberry pi to run all necessary scripts on boot. You can do this by replacing the file `etc/rc.local` on your raspberry pi with the [rc.local](/raspi-scripts/rc.local) file we provided, or more specifically you can add the following lines before "exit 0:" to your file:

```
sudo /home/pi/light_sensing.sh > /home/pi/light_sensing_log.log 2>&1 &
sudo /home/pi/MailingStatus.sh > /home/pi/mailing_log.log 2>&1 &
```
Once this is done, reboot your device and verify that it works. You can check whether the main script is running by using the following command: `ps aux | grep lightSense` or also the mailing script using `ps aux | grep mailytime`.

To debug for any issues, you can also check the logs that are created once the scripts are executed. These logs will show if any errors come up and will help you debug any issues. 

That's all! For help, questions, and general feedback, contact Rebecca Adaimi (rebecca.adaimi@utexas.edu)

## Reference 

BibTex Reference:

```
@article{10.1145/3448090,
author = {Adaimi, Rebecca and Yong, Howard and Thomaz, Edison},
title = {Ok Google, What Am I Doing? Acoustic Activity Recognition Bounded by Conversational Assistant Interactions},
year = {2021},
issue_date = {March 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {5},
number = {1},
url = {https://doi-org.ezproxy.lib.utexas.edu/10.1145/3448090},
doi = {10.1145/3448090},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = mar,
articleno = {2},
numpages = {24},
keywords = {Environmental Sounds, Activities of Daily Living, Conversational Assistants, Voice Assistants, Smart Environment, Smart Speaker, Google Home, Human Activity Recognition, Deep Learning, Audio Processing}
}
```
