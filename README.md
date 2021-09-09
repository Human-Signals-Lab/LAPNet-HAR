# Lifelong Adaptive Machine Learning for Sensor-based Human Activity Recognition Using Prototypical Networks

This is the research repository for Lifelong Adaptive Machine Learning for Sensor-based Human Activity Recognition Using Prototypical Networks. It contains the source code for *LAPNet-HAR* framework and all the experiments to reproduce the results in the paper.

## System Requirements

The deep learning system is written in `python 3`, specifically `pytorch`.

## Datasets

*LAPNet-HAR* is evaluated on 5 widely used publicly available HAR datasets:

|**Datasets** | **Activity Type** | **# of Sensor Channels** | **# of Classes** | **Balanced**|
|-------------|-------------------|:--------------------------:|:------------------:|:-------------:|
|[Opportunity](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition) | Daily Gestures | 113 | 17 | &#x2715;|
|[PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) | Physical Activities | 52 | 12 | &#x2715;|
|[DSADS](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities) | Daily & Sports Activities | 45 | 19 | &#x2713;|
|[Skoda](http://har-dataset.org/doku.php?id=wiki:dataset) | Car Maintenance Gestures | 30 | 10 | &#x2715;|
|[HAPT](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions) | Daily Activities & Postural Transitions | 6 | 12 | &#x2715;|

For every dataset, make sure they are loaded into 

That's all! For help, questions, and general feedback, contact Rebecca Adaimi (rebecca.adaimi@utexas.edu)

## Reference 

BibTex Reference:

```

```
