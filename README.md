# **Behavioral Cloning** 

## This project is to train model to steer the simulation vehicle to be on right track. input to the model measurement is steering angle. The vehicle images are collected alongside the steering angle turn.  
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior (steering angle, image)
* read data and pre-processing
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/dropout_0_2_epoch_3.png "epoch_3"
[image5]: ./examples/dropout_0_2_epoch_7.png "epoch_7"
[image6]: ./examples/run2_center_ex1.jpg "center_lane_Driving"
[image7]: ./examples/dropout_0_2_epoch_5.png "model mean squared error loss"
[image8]: ./examples/overfitting_error1.png "overfitting"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_r0.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* run4.mp4 video clip on autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run4
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24,36,48,64,64. 
The model includes RELU layers to introduce nonlinearity.

```sh

model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))

```

The data is normalized in the model using a Keras lambda layer.

```sh

model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
```

CNN network uses flatten, then engage 3 fully engaged layers (100/50/10). 

```sh

model.add(Flatten())


model.add(Dense(100))

model.add(Dropout(0.2)) # overfitting
model.add(Dense(50))

model.add(Dropout(0.2)) # overfitting
model.add(Dense(10))

model.add(Dropout(0.2)) # overfitting
model.add(Dense(1))

```


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

```sh
model.add(Dropout(0.2)) # overfitting
```

The model was trained and validated on different data sets to ensure that the model was not overfitting.

```sh
batch_size=32
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
```
![model mean squared error loss][image7]


The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
```sh
model.compile(loss='mse',optimizer='adam')
```
#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to [Nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 

The reason why this model architecture was used is, it might be appropriate as it's proven in the real world testing. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

![overfitting][image8]

To combat the overfitting, I modified the model so that "drop out" strategy was applied:
```sh
model.add(Dropout(0.2)) # overfitting
```
Then the overfitting was improved. 

![after drop out][image7]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.




#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![centerlane_driving][image6]



Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles For example, here is the code for flip image augmentation:

```sh
            augmented_images,augmented_measurements = [],[]
            for image, measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

```

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

```sh

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

```

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

```sh

```

![after drop out epochs=3][image4]

![after drop out epochs=5][image7]

![after drop out epochs=7][image5]

