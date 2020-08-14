# AI Vision Solution Kit Samples

This repository contains example notebooks to train and compile deep learning models using Amazon Sagemaker for ai-vision-solution-kit.

# Examples

The examples demonstrates you training followed by deep learning model optimization for an edge device. Similarly, the examples provides you a gentle introduction to Sagemaker functionalities. Amazon Sagemaker facilitates training and compiling models with different algorithms and frameworks. Using Amazon Sagemaker Python SDK, we can write code to train MXNet, Tensorflow, Pytorch framework models. Similarly, the examples explain about the EC2 instances for the training task and the cloud storage service S3 bucket which provides better performance thus making the training task easier for a user.

## Sagemaker Neo Compilation Jobs for Jetson Nano

Sagemaker Neo compiles the trained machine learning model to obtain an optimal performance on the target device.

### Train object detection models

The notebook samples contain fine-tuning pre-trained object detection models. Then the trained models are optimized for Jetson Nano. 

* Mask Detection SSD Mobilenet GluonCV - The example describes finetuning an object detection model that predicts the presence of a mask. Single Shot multibox Detection(SSD) with MobilenetV1.0 is used for object detection.

* Office Items Detection SSD Mobilenet GluonCV - Finetune a GluonCV SSD MobilenetV1.0 model to predict the office items such as Computer keyboard, Computer mouse, Computer monitor, Laptop, Desk and Chair.

### Compile an existing model 

Examples that provide a gentle introduction to compiling existing models for Jetson Nano.

* [Person Detection SSD Mobilenet GluonCV](https://gitlab.com/edge-vision-software/experiments/solkit-examples-test/-/tree/feature/samples-jetson-nano/sagemaker_neo_compilation_jobs_jetson_nano/compile_existing_models/ssd_person_detection_gluoncv) - This example focuses on compiling an existing model in GluonCV. The model detects persons in an image. 
 
# Getting started
For beginners, please refer to the setup guide for ai-solution-kit, training and compiling model for the target device and finally deploying the model on the edge device [AI Vision Solution Kit](https://imaginghub.com/projects/438-ai-vision-solution-kit-with-aws-cloud-connection).


# Additional information
* [Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
* [Sagemaker Neo](https://docs.aws.amazon.com/sagemaker/latest/dg/neo.html)
* [Training models in Amazon Sagemaker](https://sagemaker.readthedocs.io/en/stable/)


