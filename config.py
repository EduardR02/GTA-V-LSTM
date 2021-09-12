width = 240
height = 180
lr = 2e-4
sequence_len = 30
color_channels = 3
output_classes = 7
epochs = 5
allowed_ram_mb = 20000      # defines in how many parts the dataset will be split
random_file_order_train = True
BATCH_SIZE = 32
fps_at_recording_time = 80      # check by using main with fps only set to true, while having the game running
fps_at_test_time = 6    # check by running model in main
monitor = {'top': 27, 'left': 0, 'width': 800, 'height': 600}
outputs = {"w": 0, "a": 1, "s": 2, "d": 3, "wa": 4, "wd": 5, "nothing": 6}
amt_remove_after_pause = 300
known_normalize_growth = 4      # from uint8 to float32 exactly 4x increase

data_dir_name = "data/"
model_dir_name = "models/"
cnn_only_name = model_dir_name + "car_inception_only_4"
model_name = model_dir_name + "car_inception_pretrained_lstm_v1_cc_epoch_2"
load_data_name = data_dir_name + "training_data_for_lstm_rgb_full.npy"
temp_data_chunk_name = "temp_dataset_chunk_"
temp_data_folder_name = "data_in_chunks_temp"
new_data_dir_name = data_dir_name + "new_data/"
data_name = f"{width}x{height}_rgb"
old_data_name = "training_data_rgb_part"

