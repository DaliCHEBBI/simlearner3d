python run.py task.task_name="fit" \
datamodule.hdf5_file_path="/media/mohamedali/DUBLIN_DENSE_TRAINING_1/Stereo/dublin.hdf5" \
experiment="MSAFFDebug" \
datamodule.masq_divider=255 \
model.false1=1 \
model.false2=5 \
logger=tensorboard
