python3 tools/infer/predict_system.py \
    --use_gpu=True \
    --det_algorithm="SAST" \
    --det_model_dir="./inference/SAST" \
    --rec_model_dir="./inference/ppocr_v3" \
    --rec_image_shape="3, 48, 320" \
    --image_dir=./train_data/vietnamese/test_images/im1201.jpg \
    --drop_score=0.7 \
    --vis_font_path="./doc/fonts/vietnam-light.ttf" \
    --rec_char_dict_path="./ppocr/utils/dict/vi_vietnam.txt"