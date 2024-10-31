
DATASET_DIR="/media/mohamedali/GEOMAKER/Toronto-stereo_echo/densified_data/"

python simlearner3d/processing/dataset/toy_dataset.py +task.task_name="prepare_dataset" \
prepare_dataset.left="${DATASET_DIR}toronto_left.txt" \
prepare_dataset.right="${DATASET_DIR}toronto_right.txt" \
prepare_dataset.disp="${DATASET_DIR}toronto_disp.txt" \
prepare_dataset.masq="${DATASET_DIR}toronto_masq.txt" \
prepare_dataset.out_dir="${DATASET_DIR}Stereo"

