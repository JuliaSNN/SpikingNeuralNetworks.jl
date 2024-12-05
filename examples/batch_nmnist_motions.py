"""

NMNIST_Motions dataset class

Provides access to the event-based motions NMIST dataset. This is a version of the
NMNIST dataset in which we have separated out the events by motion and linked them
to the specified MNIST images. This allows us to retrieve any motion from the dataset
along with the associated mnist image.

"""
from os.path import exists

#import h5py
import numpy as np
import tonic
#from numpy.lib.recfunctions import merge_arrays


class NMNIST(object):
    """ A Dataset interface to the NMNIST dataset """

    def __init__(self, dataset_path, train=True, first_saccade_only=False,transform=None):
        """ Creates a dataset instance using the specified path """
        super(NMNIST, self).__init__()

        # Validate the specified path and store it
        if not exists(dataset_path):
            raise Exception("Specified dataset path does not exist")
        self._dataset = tonic.datasets.NMNIST(save_to='./data', train=train, first_saccade_only=first_saccade_only,transform=transform)
        # self.transform = transform
    """
    def get_dataset_item(self, indices):
  
        assert(len(indices) <= 100)
        all_events = []

        for id,index in enumerate(indices):
            events, label = self._dataset[index]
            label_array = np.full(events['x'].shape[0],label,dtype=[('label','i8')])
            event_array = merge_arrays((events,label_array),flatten=True)
            all_events.append(event_array)
        
        super_events = np.hstack(all_events)
        super_events = super_events[super_events['t'].argsort()]
        return super_events
		
    def get_count(self):
        """ Returns the number of  items """
        return len(self._dataset)

    def get_label_range(self):
        """ Returns the range of labels in this dataset """
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_element_dimensions(self):
        """ Returns a tuple containing the dimensions of each image in the dataset """
        return self._dataset.sensor_size
    """