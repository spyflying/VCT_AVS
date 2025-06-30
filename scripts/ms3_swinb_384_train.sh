dataset_root=${2:-"AVS_dataset/Multi-sources/"}
export DETECTRON2_DATASETS=$dataset_root

python train_net.py \
    --num-gpus 2 \
    --config-file configs/ms3_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772 \

