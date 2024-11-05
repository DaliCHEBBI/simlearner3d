
DATASET_DIR="path/to/dataset/where/left/right/disp/masq/files/exist/"


python simlearner3d/processing/dataset/toy_dataset.py +task.task_name="prepare_dataset" \
prepare_dataset.left="${DATASET_DIR}left_image_full_names.txt" \
prepare_dataset.right="${DATASET_DIR}right_image_full_names.txt" \
prepare_dataset.disp="${DATASET_DIR}disparity_images_full_names.txt" \
prepare_dataset.masq="${DATASET_DIR}occlusion_masq_image_full_names.txt" \
prepare_dataset.out_dir="${DATASET_DIR}Stereo"

