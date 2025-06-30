dataset_root=${2:-"AVS_dataset/Multi-sources/"}
export DETECTRON2_DATASETS=$dataset_root

python pred.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772 \
    --eval-only

