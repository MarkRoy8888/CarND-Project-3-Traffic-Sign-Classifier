# **Traffic Sign Recognition** 

## Writeup

## Mark's project 3 Traffic_Sign_Classifier
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:


*Number of training examples = 34799
*Number of testing examples = 12630
*Image data shape = (32, 32, 3)
*Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

1.I just process the image normalize the images for learning speed.
2.I didn't grayscaling, because in my opinion the sign's color is useful.
3.Check the distributed of train, vaild and test. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_15 (Conv2D)           (None, 32, 32, 64)        4864      
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 16, 16, 128)       73856     
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 16, 16, 128)       147584    
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 8, 8, 256)         295168    
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 8, 8, 256)         590080    
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 4, 4, 256)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_9 (Dense)              (None, 128)               524416    
_________________________________________________________________
dropout_6 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 64)                8256      
_________________________________________________________________
dropout_7 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_11 (Dense)             (None, 43)                2795      
=================================================================
Total params: 1,647,019
Trainable params: 1,647,019
Non-trainable params: 0
_________________________________________________________________


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* I use the a minst model sample as base.
* And change some parameter. 
* the result is ok for me.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


the best result

Epoch 19/20
1088/1088 [==============================] - 122s 112ms/step - loss: 0.0615 - accuracy: 0.9860 - val_loss: 0.7114 - val_accuracy: 0.9512

the test result

395/395 [==============================] - 9s 24ms/step - loss: 0.4717 - accuracy: 0.9162 Test Accuracy = 0.916


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

1/1 [==============================] - 0s 135ms/step - loss: 10.6423 - accuracy: 0.4000  Test Accuracy = 0.400

![alt-text-1](out_sign/images_1.png "images_1")
![alt-text-1](out_sign/images_2.png "images_2")
![alt-text-1](out_sign/images_3.png "images_3")
![alt-text-1](out_sign/images_4.png "images_4")
![alt-text-1](out_sign/images_5.png "images_5")

Before learning I think the speed limit 60 is hard to told between 50 and 60, and image3 is difficult to distinguish from many sign.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

![alt-text-1](readme/image1_result.png "images_1")
![alt-text-1](readme/image2_result.png "images_2")
![alt-text-1](readme/image3_result.png "images_3")
![alt-text-1](readme/image4_result.png "images_4")
![alt-text-1](readme/image5_result.png "images_5")

the new sign will different from country, incluing color, shape, type and content, so the result is not good.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)





