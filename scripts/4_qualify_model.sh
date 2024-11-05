python run.py task.task_name="qualify" \
datamodule.hdf5_file_path="/media/.../DUBLIN_DENSE_TRAINING_1/Stereo/dublin.hdf5" \
datamodule.masq_divider=255 \
experiment="SIMDebug" \
model="msaff_model" \
model.ckpt_path="/home/.../opt/simlearner3d/tests/logs/runs/2024-10-30/09-57-37/checkpoints/epoch_007.ckpt" \
+report.output_folder="."
