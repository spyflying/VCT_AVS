dataset_root=${2:-"AVS_dataset/AVSBench_semantic/"}
export DETECTRON2_DATASETS=$dataset_root

python train.py \
    --num-gpus 4 \
    --config-file configs/ss_swinb_bs8_45k_384/COMBO_SWINB.yaml \
    --resume \
    --dist-url tcp://0.0.0.0:47772