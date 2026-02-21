# python eval.py \
#     --test_file /home/zyy/Semantic_change_detection/dataset/SECOND/test.txt \
#     --prediction_dir Trained_model/SECOND/test \
#     --label_dir /home/zyy/SCD/datasets/SECOND/reorganized/label \
#     --num_classes 30

# python eval.py \
#     --test_file /home/zyy/SCD/datasets/JL1_2023/reorganized/list/test.txt \
#     --prediction_dir Trained_model/JL1/test \
#     --label_dir /home/zyy/SCD/datasets/JL1_2023/reorganized/label \
#     --num_classes 9

# python eval.py \
#     --test_file /home/zyy/SCD/datasets/FZ_SCD_processed/list/test.txt \
#     --prediction_dir Trained_model/FZSCD/test \
#     --label_dir /home/zyy/SCD/datasets/FZ_SCD_processed/label \
#     --num_classes 6

python eval.py \
    --test_file /home/zyy/SCD/datasets/CropSCD/original_split/test.txt \
    --prediction_dir Trained_model/CropSCD/test \
    --label_dir /home/zyy/SCD/datasets/CropSCD/label \
    --num_classes 9