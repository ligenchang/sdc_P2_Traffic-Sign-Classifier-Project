#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/Visualization.JPG "Visualization"
[image2]: ./img/grayscale.jpg "Grayscaling"
[image3]: ./img/augmented.png "Augmented"
[image4]: ./img/newimage.JPG "newimage"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set

* The size of training set is 34799 (X_train.shape[0])
* The size of the validation set is 4410(X_valid.shape[0])
* The size of test set is 12630 ((X_test.shape[0]))
* The shape of a traffic sign image is 32 * 32 * 3 (X_train[0].shape)
* The number of unique classes/labels in the data set is 43 (np.unique(np.concatenate([y_train, y_test, y_valid])).shape[0])

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data conposition for each unique labels. From the bar chart, we can see that the data for different labels which are not well balanced and some categories have more than 2000 samples and some has less than 200.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it shows the evendience that the color image won't help to increase the accuracy for the training model.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


Nest step is to increase the picture contract as some of the images's contract is too low and can't see clearly. To make the image more clear and also don't lose details, I use the localized histogram_equalize to increase the contrast.

As a last step, I normalized the image data because it will make the data has 0 mean and will reduce the impact of a suden big number.


Then I decided to generate additional data because we need more training data to simulate different real scanrios which doesn't contain in the original training set. 

To add more data to the the data set, I used the following techniques because tranform the image will make the CNN learn more complex situations. 

Firstly, I decided to tranform the original images and add it to the training set so that i could have more data and balanced data samples we can use for training.

Here is an example of a traffic sign image before and after tranform.



After that, i just roate the images with degree from -15 to 15 to create more images.

Then I concatenate the orignal image, the tranoformed images and the rotated images to re-form the training set. 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following:
1.Image rotation
2.Image transformation
3.Image shear 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x36 	|
| RELU					| tf.nn.relu activation function to make it non-linear											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x36				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 28x28x56     									|
| Fully connected		| Input 900, output 120       
| Fully connected		| Input 120, output 84     
| Fully connected		| Input 84, output 60 
| Fully connected		| Input 60, output 43 
| Softmax				| tf.nn.softmax_cross_entropy_with_logits.        									|
|						|												|
|						|												|
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an tf.train.AdamOptimizer to minimize the the mean of cross_entropy. The batch size is 128 and epochs used is 20. The learning rate is 0.001.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 99.1%
* test set accuracy of 96.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I choose lenet as my initial arhitecture and I also referred to AlexNet architecture and tried to increase the convolution layer depth according to AlexNet. 
* What were some problems with the initial architecture?
i find the convetional layer depth was 6 but i find that that's not enough so that i increased it and tried to find a better value for that.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The depth of convolutional layer1 and layer2 were adjusted because the validation will be low if the depth is 6 and 16, the new value is 26 and 36. I also add one more layer for the full connect.
* Which parameters were tuned? How were they adjusted and why?
The depth of convolutional layer1 and layer2 were adjusted because the validation will be low if the depth is 6 and 16, the new value is 26 and 36. To increase the convolution layer depth will enable the network to capture more feature maps so that it will predicate more precisely.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The keep prob of drop out layer was 75%, but i realized one issue later, when i used the model to test the new image, the test result may be different for each session. Soon I realized it's due to the drop-out issue in the model, then I remove it to keep the model stable.


If a well known architecture was chosen:
* What architecture was chosen?
Le-Net
* Why did you believe it would be relevant to the traffic sign application?
Le-Net was invented in 1998 and was proven to be very effitive for CNN.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Accuracy for training set: 99.8% (As the training set is too big so than I ramdomly sampled 0.5% data to calculate the training set accuracy)
Accuracy for validation set: 99.1%
Accuracy for Test Set: 96.9%
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

The third image might be difficult to classify because this image was stretched and not in it's original percentage.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road 12    		| Priority road  12									| 
| Road work 25    			| Road work 25 										|
| STOP 14					| Priority road 12	 										|
| Speed limit (30km/h)	1   | Speed limit (30km/h) 	 1			 				|
| Bicycles crossing 29	    | Bicycles crossing 29     							|




The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.9%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.36), and the image does contain a proper priority sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .79        			| Priority road 12   									| 
| .28     				| Road work 25										|
| .19					| Priority road 12											|
| .50	      			| Speed limit (30km/h)	1				 				|
| .26				    | Bicycles crossing 29    							|


For the third image, it is a stop sign, but it was stretched so that height and weight is not in it's orignal constraint ratio. It's recognized as Priority road instead of STOP unfortunatelly. However, the probability of the first 2 are quite close, it's 19.44841766,  19.34706116. the stop sign is the 19.34706116 probablity.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I visualize the first new image, but from the feature map, i can't understand what the CNN is trying to understand.


