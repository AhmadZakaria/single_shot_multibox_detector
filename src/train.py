import os

from keras.callbacks import CSVLogger
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import SGD

from datasets import CSVDataManager
from models import SSD300
from models.experimental_loss import MultiboxLoss
from utils.boxes import create_prior_boxes
from utils.boxes import to_point_form
# from utils.generator import ImageGenerator
from utils.sequencer_manager import SequenceManager
from utils.training_utils import LearningRateManager

# hyper-parameters
batch_size = 32
num_epochs = 233
image_shape = (300, 300, 3)
box_scale_factors = [.1, .1, .2, .2]
negative_positive_ratio = 3
learning_rate = 1e-4
weight_decay = 5e-4
momentum = .9
optimizer = SGD(learning_rate, momentum, decay=weight_decay)
# optimizer = 'adam'
decay = 0.1
step_epochs = [154, 193, 232]
randomize_top = True
weights_path = '../trained_models/SSD300_weights.hdf5'
train_datasets = ['roof01', 'drone_wide01']
train_splits = ['trainval', 'train']
val_dataset = 'drone_wide01'
val_split = 'val'
class_names = [0, 1]
difficult_boxes = True
model_path = '../trained_models/SSD_SGD_scratch_all2/'
save_path = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

train_data_manager = CSVDataManager(dataset_name=train_datasets, split=train_splits, class_names=class_names)
train_data = train_data_manager.load_data()
class_names = train_data_manager.class_names
num_classes = len(class_names)
exit(0)
val_data_manager = CSVDataManager(val_dataset, val_split, class_names, False)
val_data = val_data_manager.load_data()

# generator
prior_boxes = to_point_form(create_prior_boxes())
train_sequencer = SequenceManager(train_data, 'train', prior_boxes,
                                  batch_size, box_scale_factors, num_classes)

val_sequencer = SequenceManager(val_data, 'val', prior_boxes,
                                batch_size, box_scale_factors, num_classes)

# model
multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio, batch_size)
model = SSD300(image_shape, num_classes, weights_path)
model.compile(optimizer, loss=multibox_loss.compute_loss)

# callbacks
if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoint = ModelCheckpoint(save_path, verbose=1, period=1)
log = CSVLogger(model_path + 'SSD_scratch.log')
learning_rate_manager = LearningRateManager(learning_rate, decay, step_epochs)
learning_rate_schedule = LearningRateScheduler(learning_rate_manager.schedule)
tbCallBack = TensorBoard(log_dir='./Graph/', histogram_freq=0, batch_size=batch_size, write_graph=True)

callbacks = [checkpoint, log, learning_rate_schedule, tbCallBack]

# model fit
model.fit_generator(train_sequencer,
                    steps_per_epoch=int(len(train_data) / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_sequencer,
                    validation_steps=int(len(val_data) / batch_size),
                    use_multiprocessing=False,
                    max_queue_size=70,
                    workers=5)
