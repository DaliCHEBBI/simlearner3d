python run.py task.task_name="qualify" \
datamodule.hdf5_file_path="/home/.../Echantillon_dublin/Stereo/dublin.hdf5" \
datamodule.masq_divider=255 \
datamodule.patch_size=512 \
trainer.accelerator="cpu" \
experiment="SIMDebug" \
model="msaff_model" \
model.ckpt_path="/home/.../Echantillon_dublin/10-46-30/checkpoints/epoch_014.ckpt" \
+report.output_folder="."
