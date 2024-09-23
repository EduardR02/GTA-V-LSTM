import os

width = 240
height = 180
lr = 2e-4
cnn_lr = 2e-5
sequence_len = 2
sequence_stride = 20     # Period between successive output sequences
color_channels = 3
output_classes = 6     # was 11
counts_as_tap = 0.4
epochs = 30
allowed_ram_mb = 12000      # defines in how many parts the dataset will be split
random_file_order_train = True
BATCH_SIZE = 64
CNN_ONLY_BATCH_SIZE = 256
fps_at_recording_time = 80      # check by using main with fps only set to true, while having the game running
fps_at_test_time = 5    # check by running model in main
monitor = {'top': 27, 'left': 0, 'width': 800, 'height': 600}
outputs = {"w": 0, "a": 1, "s": 2, "d": 3, "wa": 4, "wd": 5, "nothing": 6}
outputs_base = {"w": 0, "a": 1, "s": 2, "d": 3, "nothing": 4}
amt_remove_after_pause = 300

data_dir_name = "data"
model_dir_name = "models"
lstm_dir = os.path.join(model_dir_name, "lstm")
cnn_dir = os.path.join(model_dir_name, "cnn")
ckpt_name_cnn = "ckpt.pt"
ckpt_name_rnn = "ckpt.pt"
new_data_dir_name = os.path.join(data_dir_name, "new_data")
turns_data_dir_name = os.path.join(data_dir_name, "turns")
stuck_data_dir_name = os.path.join(data_dir_name, "stuck")
back_on_road_data_dir_name = os.path.join(data_dir_name, "back_on_road")
data_name = f"{width}x{height}_rgb"
