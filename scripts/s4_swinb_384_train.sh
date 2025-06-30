dataset_root=${2:-"AVS_dataset/Single-source/"}
export DETECTRON2_DATASETS=$dataset_root

python train_net.py \
    --num-gpus 4 \
    --config-file configs/s4_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772
    