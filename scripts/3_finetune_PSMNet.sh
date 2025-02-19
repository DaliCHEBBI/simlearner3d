python run.py task.task_name="finetune" \
datamodule.hdf5_file_path="/var/data/MAChebbi/dublin.hdf5" \
datamodule.masq_divider=255 \
datamodule.patch_size=512 \
datamodule.batch_size=2 \
experiment="PSMNETDebug" \
trainer.accelerator="gpu" \
++trainer.accumulate_grad_batches=4 \
++trainer.precision="16-mixed" \
model="psmnet" \
++model.load_pretrained=true \
+model.ckpt_path="/home/MAChebbi/repositories/simlearner3d/simlearner3d/models/PSMNet/models/trained_assets/finetune_PSMnet.tar" \
++model.channel=3 \
logger=comet \
++logger.comet.experiment_name="test_psmnet_normalization_finetune"


#++model.load_pretrained=true \
#+model.ckpt_path="/home/MAChebbi/repositories/simlearner3d/simlearner3d/models/PSMNet/models/trained_assets/finetune_PSMnet.tar" \
