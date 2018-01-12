import csv
import os

import numpy as np

from .data_utils import get_class_names


class CSVDataManager(object):
    def __init__(self, dataset_name='drone_wide01', split='train',
                 class_names='all', datasets_root_path='../datasets/hs_datasets/',
                 input_format=['image_name', 'class_id', 'xmin', 'xmax', 'ymin', 'ymax'],
                 box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'],
                 size=(2048, 1088)):

        self.dataset_name = dataset_name
        self.split = split
        self.class_names = class_names
        if class_names == 'all':
            self.class_names = get_class_names(self.dataset_name)
        self.datasets_root_path = datasets_root_path
        self.images_path = None
        self.arg_to_class = None
        self.w, self.h = size
        if input_format is not None:
            self.input_format = input_format
        if box_output_format is not None:
            self.box_output_format = box_output_format

        self.filenames = []
        self.labels = []

    def load_csv(self, dataset_name=None, split=None):
        if split is None:
            split = self.split
        if dataset_name is None:
            dataset_name = self.dataset_name

        dataset_path = os.path.join(self.datasets_root_path, dataset_name)

        # First, just read in the CSV file lines and sort them.
        data = []

        with open(os.path.join(dataset_path, split + ".txt"), newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread)  # Skip the header row.
            for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
                # If the class_id is among the classes that are to be included in the dataset...
                if self.class_names == 'all' or int(row[self.input_format.index(
                        'class_id')].strip()) in self.class_names:
                    # Store the box class and coordinates here
                    box = [row[self.input_format.index('image_name')].strip()]
                    # For each element in the output format
                    #  (where the elements are the class ID and the four box coordinates)...
                    for element in self.box_output_format:
                        # ...select the respective column in the input format and append it to `box`.
                        element_val = int(row[self.input_format.index(element)].strip())
                        if element in ['xmin', 'xmax']: element_val /= self.w
                        if element in ['ymin', 'ymax']: element_val /= self.h
                        box.append(element_val)
                    data.append(box)

        data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result
        print("data: ", len(data))
        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = data[0][0]  # The current image for which we're collecting the ground truth boxes
        current_labels = []  # The list where we collect all ground truth boxes for a given image
        for i, box in enumerate(data):

            if box[0] == current_file:  # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(dataset_path, current_file))
            else:  # If this box belongs to a new image file
                self.labels.append(np.stack(current_labels, axis=0))
                self.filenames.append(os.path.join(dataset_path, current_file))
                current_labels = []  # Reset the labels list because this is a new file.
                current_file = box[0]
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(dataset_path, current_file))

    def convert_data_to_dict(self):
        print(len(self.filenames), len(self.labels))
        print(self.filenames[0], self.labels[0])

    def load_data(self):

        if isinstance(self.dataset_name, list):
            if not isinstance(self.split, list):
                raise Exception("'split' should also be a list")
            for ds, split in zip(self.dataset_name, self.split):
                self.load_csv(ds, split)
        else:
            self.load_csv(self.dataset_name)

        self.convert_data_to_dict()

        # return ground_truth_data

    def _to_one_hot(self, idx):
        one_hot_vector = [0] * len(self.class_names)
        one_hot_vector[idx] = 1
        return one_hot_vector
