# test obama eo
python test.py \
    --pose data/obama.json \
    --ckpt pretrained/obama_eo.pth \
    --aud data/intro_eo.npy \
    --workspace trial_test \
    --bg_img data/bg.jpg \
    -O --torso --data_range 0 100

# merge audio with video
ffmpeg -y -i trial_test/results/ngp_ep0028.mp4 -i data/intro.wav -c:v copy -c:a aac obama_eo_intro.mp4

# # test obama ds
# python test.py \
#     --pose data/obama.json \
#     --ckpt pretrained/obama.pth \
#     --aud data/intro.npy \
#     --workspace trial_test \
#     --bg_img data/bg.jpg \
#     -O --torso --data_range 0 100 --asr_model deepspeech

# # merge audio with video
# ffmpeg -y -i trial_test/results/ngp_ep0056.mp4 -i data/intro.wav -c:v copy -c:a aac obama_intro.mp4