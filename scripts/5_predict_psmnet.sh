python run.py task.task_name="predict" \
model="psmnet" \
++model.channel=3 \
++model.load_pretrained=false \
+predict.ckpt_path="/home/mohamedali/opt/simlearner3d/simlearner3d/models/PSMNet/models/trained_assets/finetune_PSMnet.tar" \
+predict.left_image="" \
+predict.right_image="/home/MAChebbi/repositories/simlearner3d/tests/data/DMTrain_DUBL-3489_DUBLIN_AREA_2KM2_rgb_124890_id283c1_20150326120956_3489_DUBLIN_AREA_2KM2_rgb_124891_id284c1_20150326120958_0020_Im2.tif" \
+predict.gpus=1 \
+predict.output_dir="/home/MAChebbi/repositories/simlearner3d/tests" \
++predict.disp_scale=1

#+predict.ckpt_path="/home/mohamedali/opt/simlearner3d/simlearner3d/models/PSMNet/models/trained_assets/finetune_PSMnet.tar" \
#+predict.ckpt_path="/home/MAChebbi/opt/simlearner3d/tests/logs/runs/2025-02-19/10-30-42/checkpoints/epoch_010.ckpt" \
#+predict.ckpt_path="/home/MAChebbi/opt/simlearner3d/tests/logs/runs/2025-02-18/10-59-39/checkpoints/epoch_008.ckpt" \

#/home/MAChebbi/opt/simlearner3d/tests/logs/runs/2025-02-18/18-52-35/checkpoints/epoch_018.ckpt
