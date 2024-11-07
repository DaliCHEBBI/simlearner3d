
DATASET_DIR="/home/.../Echantillon_dublin/"


python simlearner3d/processing/dataset/toy_dataset.py +task.task_name="prepare_dataset" \
prepare_dataset.left="${DATASET_DIR}l_1_left_train.txt" \
prepare_dataset.right="${DATASET_DIR}l_1_right_train.txt" \
prepare_dataset.disp="${DATASET_DIR}l_1_disp_train.txt" \
prepare_dataset.masq="${DATASET_DIR}l_1_masq_train.txt" \
prepare_dataset.out_dir="${DATASET_DIR}Stereo" \
data_split.train=95 \
data_split.val=2 \
data_split.test=3


