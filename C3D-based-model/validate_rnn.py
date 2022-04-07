from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet
import tensorflow as tf
import os
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def validate(data_type, model, seq_length=24, saved_model=None,
             class_limit=None, image_shape=None,
             val_steps=None):
    batch_size = 1

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )
    
    sample_name = []
    train_sample,test_sample = data.split_train_test ()
    label = []
    for i in range(0,len(test_sample)):
        label.append(test_sample[i][1])
        sample_name.append(test_sample[i][2])
    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    
    results_predict = rm.model.predict_generator(
        generator=val_generator,
        val_samples=val_steps) 
    results_evaluate = rm.model.evaluate_generator(
        generator=val_generator,
        val_samples=val_steps)
    print('-'*80)
    
    y_pre = []
    for i in range (0,val_steps):
        print('Sample %2d, the predicted probability is %2.2f%% ' % (i+1,results_predict[i][1] * 100))
        y_pre.append(results_predict[i][1])

    print('-'*80)

    y_label = []  
    for j in range(0,len(label)):
        y_label.append(int(label[j]))
    fpr, tpr, thersholds = roc_curve(y_label, y_pre, drop_intermediate=False)
 
    roc_auc = auc(fpr, tpr)
    print('The AUC of this model is : ',roc_auc)    


def main():
    model = 'c3d'

    saved_model = './data/ckpt/c3d-based-model.hdf5'  
    data_type = 'images'
    image_shape = (392, 392, 3)
    
    validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=None,
             val_steps=30)

if __name__ == '__main__':
    main()
