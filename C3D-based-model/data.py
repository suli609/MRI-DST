"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from processor import process_image,process_image_aug
from keras.utils import to_categorical
import tensorflow as tf
from keras_preprocessing import image
import random


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test



    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))
        i=0
        while 1:
            X, X_num, y = [], [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
#                sequence = None

                # Get a random sample.
                if  train_test == 'train' :
                    sample = random.choice(data)
                    # sample = data[i]
                    # if i < (len(data) -1):
                    #     i=i+1
                    # else:
                    #     i = 0
                else:
                    sample = data[i]
                    if i < (len(data) -1):
                        i=i+1
                    else:
                        i = 0
                    #i =  i + data.count(sample)
#                if  train_test == 'test' :
                #    data.remove(sample)
#                    if data==[]:
#                       print("a epoch data(%s) has been test" %i)
                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    if train_test == 'train':
                    # Get and resample frames.
                        frames = self.get_frames_for_sample(sample)  #one sequence
                        frames = self.rescale_list(frames, self.seq_length)
                        if frames == []:
                            print(sample[2])
                    # Build the image sequence
                        if len(frames) >=self.seq_length:
                            sequence = self.build_image_sequence_aug(frames)
                        else:
                            sequence = sequence
                    #    sequence = self.build_image_sequence(frames)
                    else:
                        frames = self.get_frames_for_sample(sample)  #one sequence
                        frames = self.rescale_list(frames, self.seq_length)
                        # Build the image sequence
                        sequence = self.build_image_sequence(frames)                   
                    num_data = [sample[4],sample[5],sample[6],sample[7],sample[8],sample[9]]

                
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                
                X.append(sequence)
                X_num.append(num_data)
                y.append(self.get_class_one_hot(sample[1]))

        #    yield np.array(X),np.array(y)
            yield [np.array(X),np.array(X_num)], np.array(y)
        #    yield np.array([X,X_num]), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]
    
    def build_image_sequence_aug(self, frames,):
        """Given a set of frames (filenames), build our sequence."""
        h_flip=random.randint(1,100)
        v_flip=random.randint(1,100)
        rot=random.randint(-90,90)
        x_shift = random.uniform(0, 100)
        y_shift = random.uniform(0, 100)
        lighting_k = random.uniform(1/4,4)
        lighting_b = random.uniform(0,10)
        G_noise=random.randint(1,100)
        
        return [process_image_aug(x, self.image_shape,h_flip,v_flip,rot,x_shift,y_shift,lighting_k,lighting_b,G_noise) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join('data', sample[0], sample[1])
        #filename = '*.jpg'
        #images = sorted(glob.glob(os.path.join(path, filename )))
        filename = sample[2]  
        images = sorted(glob.glob(os.path.join(path, filename +'/' + '*jpg')))
       # print (images)
        return images
    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        if len(input_list) < size :
            print('input_list',input_list)
            return []
            
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
