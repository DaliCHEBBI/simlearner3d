python run.py task.task_name="predict" \
model="psmnet" \
+predict.ckpt_path="/home/mohamedali/opt/simlearner3d/simlearner3d/models/PSMNet/models/trained_assets/finetune_PSMnet.tar" \
+predict.left_image="/home/mohamedali/opt/simlearner3d/tests/Stereo/train/DMTrain_ENSH-0021764_1_0021765_1_0000_Im1.tif" \
+predict.right_image="/home/mohamedali/opt/simlearner3d/tests/Stereo/train/DMTrain_ENSH-0021764_1_0021765_1_0000_Im2.tif" \
+predict.gpus=0 \
+predict.output_dir="/home/mohamedali/opt/simlearner3d/tests" \
+predict.disp_scale=256
