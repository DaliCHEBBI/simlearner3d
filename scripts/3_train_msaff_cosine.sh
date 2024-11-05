python run.py task.task_name="fit" \
datamodule.hdf5_file_path="/media/.../DUBLIN_DENSE_TRAINING_1/Stereo/dublin.hdf5" \
datamodule.masq_divider=255 \
experiment="SIMDebug" \
model="msaff_model" \
++model.mode="feature" \
++logger.comet.experiment_name="msaff_cos"
