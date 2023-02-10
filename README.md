# Common Repository

This repository contains classes and functions that are called in multiple notebook files.

Important Files/Folder in this repo:
  * main.py: Following functions are present in this file
    * train: For training model on train dataset
    * test: For evaluating model on test dataset
    * train_test_model: Model is executed in this function. Optimized, scheduler, number of epochs, lambda for L1 etc are managed in this.
  * utils.py
  * models folder
    *resnet.py : It contains Resnet18 & Resnet34 definition
