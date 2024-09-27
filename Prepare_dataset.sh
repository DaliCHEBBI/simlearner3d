
DATASET_DIR="/mnt/store-lidarhd/projet-LHD/IA/MYRIA3D-SHARED-WORKSPACE/MAChebbi/DATASETS/"

python simlearner3d/processing/dataset/toy_dataset.py +task.task_name="prepare_dataset" \
prepare_dataset.left="${DATASET_DIR}eurosdr_vahingen_left_train.txt" \
prepare_dataset.right="${DATASET_DIR}eurosdr_vahingen_right_train.txt" \
prepare_dataset.disp="${DATASET_DIR}eurosdr_vahingen_disp_train.txt" \
prepare_dataset.masq="${DATASET_DIR}eurosdr_vahingen_masq_train.txt" \
prepare_dataset.out_dir="${DATASET_DIR}Stereo"

