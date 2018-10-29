# tf

This repository contains a variety of projects and attempts at several different machine learning sequence models, using TensorFlow. 

## Example
This is an path modeling algorithm used to calculate the necessary path of a robot so as to reach a specific target point and target orientation. It is calculated by simulating the robot step by step, at each step calculating the wheel speed values from a simple feed forward network. Then, I backpropagate over the entire simulation using error of the final position and orientation to adjust the weights of the neural network. The code can be found in the `robo` folder.

![Robot](https://www.github.com/fingoldin/tf/raw/master/robot.png "Robot path finding")
