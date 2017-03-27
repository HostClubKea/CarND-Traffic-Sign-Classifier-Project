#**Traffic Sign Recognition** 

This is my review on traffic sign recognition problem using convolutional newtwork

[//]: # (Image References)
[image1]: ./images/index.png "Training samples by class"
[image2]: ./images/accuracy_amount_relation.png "Relationship between accuracy and number of training examples"
[image3]: ./images/bright.png "Bright images"
[image4]: ./images/dark.png "Dark images"
[image5]: ./images/bright_adapthist.png "Bright images"
[image6]: ./images/dark_adapthist.png "Dark images"
[image7]: ./images/distribution_brightness.png "Distribution by brightness"
[image8]: ./images/original.png "Preprocessed images"
[image9]: ./images/transformed.png "Augmented images"
[image10]: ./images/augmented.png "Relationship between accuracy and number of test examples when trained on augmented dataset"
[image11]: ./images/loss.png "Loss"
[image12]: ./images/accuracy.png "Accuracy"
[image13]: ./images/new_signs.png "New signs"
[image14]: ./images/new_sign.png "New sign"
[image15]: ./images/new_sign_confidence.png "Confidence"



###Data Set Summary & Exploration

####1. Basic summary
The German Traffic Sign Dataset consists of 34,799 trainig images, 4410 validation images and 12,630 test images. Images have shape - 32x32x3 (Height x Width x RGB). Each sample is a traffic sign belonging to one of 43 classes. (**In[3]**)

####2. Dataset visualisation
Dataset is very unbalanced, and some classes are represented way better than the others. This we could see on next graph (**In[5]**)

![alt text][image1]

In early experiments with LeNet5 there was quite visible ralationship between amount of training example for the class and accuracy of this class in test dataset

![alt text][image2]


Another problem of dataset that it contains some very bright and dark images which hard to recognise without exposure adjustment. In next histogram presented distribution of images by their overall brightness(**In[7]**)

![alt text][image7]

**Brightest and darkest images**
![alt text][image3]
![alt text][image4]

###Preprocessing
After some experiments preprocessing have next steps(**In[8]**):

- Scaling of pixel values to [0, 1] (originally [0, 255]

- Applying histogram equalization

- Extracting Y channel of the YCbCr representation 

This steps was tested on simple LeNet5 model and including them one by one gave every time better results. Only with this steps model showed around 92% accuracy insted of 87% in the beginning


**Brightest and darkest images after histogram equalization**
![alt text][image5]
![alt text][image6]


###Augmentation
To generalize our model better and balance dataset we need some ways to create additional samples.


####1.Flipping(**In[9]**)
As some samples has symmetry we can easyly increase number of samples by horizontal and vertical flipping. For some of this operations we would need to change class of flipped sample


####2.Applying transformations(**In[9]**)

Number of samples could be increased also by applying zooming, rotation, shifting, sheering transformations

**Preprocessed images**
![alt text][image8]

**Augmented images**
![alt text][image9]

When model trained on augmented balanced dataset we can see much less difference between well amd previously poorly presented classes

![alt text][image10]

Augmetation is quite time consuming operation, so I generated augmented validation and training dataset and later used them for train my model

###Model

I started with simple LeNet archtecture - only change was amount of outputs. From the beginning it was clear that this model was underfitted, but still only with some data preprocessing it could get from 87% to 92% accuracy.
 
To be able to test different parameter and datasets on model I introduced parameter container (**In[11]**), so form the beginning i can change some hyperparameters and see how they change accuracy of the model. Problem is at some point checking how changing parameters change accuracy becomes very long affair - to see small change you would need to wait several hours. So at some point I decided to improve LeNet5 with multiscale features and dropouts, so at the end it have 3 convolution layers and 2 fullyconnected ones.

Final model described with parameters (**In[18]**) and model itself is build in (**In[12]**)


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x25 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Dropouts 	      	    | keep 90%      				 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x50    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x50 				    |
| Dropouts 	      	    | keep 80%      				 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 8x8x100     |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x100 				    |
| Dropouts 	      	    | keep 70%      				 				|
| Flattened				| 		       									|
| Fully connected		| 500        									|
| Relu					| 	        									|
| Dropouts				| keep 50%     									|
| Fully connected		| 200        									|
| Relu					| 	        									|
| Dropouts				| keep 50%     									|
| Output				| 43    	   									|

###Training
I took code for training from LaNet5 lab which used AdamOptimizer and was optimizing accuracy. 

First change was adding early stopping as changing amount of epochs wasn't smart solution for finding best fit. I ended up with 0.001 learning rate as it was faster for fitting model, but if there was enough time i would finetune model with 0.0001 learning rate. I have max epoch equals 500 but and stop learning if theere was no validation loss improvement in 10 epoches. If I have enough time and want better result i would also increase this value.

After some time I had moments when my validation accuracy was 100 and I switched to optimizing loss function, because we want not only predict right but be more confident.

Next step was addind l2_loss to decrease overfitting to train dataset. It should punish loss function for extremely big weights in fully connected layers. With dropouts they should prevent model to rely on small set of features

Parameters for training set there(**In[18]**) and training network is described at (**In[17]**)


###Results and Decisions
My final model results were:

- Train Accuracy = 99.994

- Validation Accuracy = 98.889

- Test Accuracy = 97.680

![alt text][image11]
![alt text][image12]


This shows that model still have tendention to overfit. Probably if I increase augmentation intensity or/and would augment some data in each epoch I might get better results but this would strongly increase training time.

Also most described architecture use more depth in their convolutional layer so it would also should improve result but also it would increase training time.



###Test a Model on New Images

####1. New sign photo
As it happenes I live in Berlin now and easiest solution was going out and take some photo myself


![alt text][image13]

[Priority road, Yield, Turn right ahead, Ahead only, General caution]

First sign I took specially covered by some object to see how well it would be recognised (**In[39]**)

####2. Predictions

All signs were predicted right [Priority road, Yield, Turn right ahead, Ahead only, General caution] (**In[40]**)


Which gives 100% accuracy.

####3. Model confidence

In (**In[43]**) we can see that model was very confident in each case

As an example "Priority road" sign:

![alt text][image14]
![alt text][image15]


