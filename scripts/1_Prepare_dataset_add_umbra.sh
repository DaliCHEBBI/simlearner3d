
DATASET_DIR="/home/MAChebbi/opt/simlearner3d/"


python simlearner3d/processing/dataset/toy_dataset.py +task.task_name="prepare_dataset" \
prepare_dataset.left="${DATASET_DIR}l_1_left_train.txt" \
prepare_dataset.right="${DATASET_DIR}l_1_right_train.txt" \
prepare_dataset.disp="${DATASET_DIR}l_1_disp_train.txt" \
prepare_dataset.masq="${DATASET_DIR}l_1_masq_train.txt" \
prepare_dataset.out_dir="/home/MAChebbi/opt/simlearner3d/" \
data_split.train=92 \
data_split.val=4 \
data_split.test=4


