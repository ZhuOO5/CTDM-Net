# python predict.py \
#     --config config_fusion/SECOND.yaml \
#     --model_dir Trained_model/SECOND/best_model.pth \
#     --n_class 30 \
#     --save_dir Trained_model/SECOND/test \
#     --img_size 512 \

# python predict.py \
#     --config config_fusion/JL1.yaml \
#     --model_dir Trained_model/JL1/best_model.pth \
#     --n_class 9 \
#     --save_dir Trained_model/JL1/test \
#     --img_size 256 \

python predict.py \
    --config config_fusion/FZSCD.yaml \
    --model_dir Trained_model/FZSCD/best_model.pth \
    --n_class 6 \
    --save_dir Trained_model/FZSCD/test \
    --img_size 256 \

# python predict.py \
#     --config config_fusion/CropSCD.yaml \
#     --model_dir Trained_model/CropSCD/best_model.pth \
#     --n_class 9 \
#     --save_dir Trained_model/CropSCD \
#     --img_size 512 \