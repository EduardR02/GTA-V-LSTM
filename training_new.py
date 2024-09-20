import numpy as np
import config
import utils
import torch
import os
from contextlib import nullcontext
from dataloader import get_dataloader
from model import Dinov2ForClassification, Dinov2ForTimeSeriesClassification
import matplotlib.pyplot as plt
import math
import time

current_data_dirs = [config.turns_data_dir_name, config.new_data_dir_name]  # has to be list


print(torch.backends.cudnn.version())
print(torch.version.cuda)
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = os.path.join('models', 'test')
eval_interval = 200
log_interval = 1
eval_iters = 8
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
fine_tune = False   # train the entire model or just the top
init_from = 'scratch' # 'scratch' or 'resume'
dino_size = "base"
checkpoint_name = "ckpt.pt"
metrics_name = "metrics_plot.png"
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 64    # if gradient_accumulation_steps > 1, this is the micro-batch size
train_split = 0.95   # test val split, important to keep it to reproduce obv
convert_to_greyscale = False
sequence_len = 1
sequence_stride = 20
classifier_type = "cce" # "cce" or "bce"

# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 60000 # total number of training iterations
# optimizer settings
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.995
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 400 # how many steps to warm up for
lr_decay_iters = 20000 # should be ~= max_iters per Chinchilla
min_lr = 5e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# change this to bf16 if your gpu actually supports it
dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_dict = {k: globals()[k] for k in config_keys} # will be useful for logging


os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if classifier_type == "bce":
    id2label = {0: "w", 1: "a", 2: "s", 3: "d"}
else:
    id2label = config.outputs


train_dataloader = get_dataloader(current_data_dirs, batch_size, train_split, True, classifier_type, sequence_len, sequence_stride, shuffle=True)
val_dataloader = get_dataloader(current_data_dirs, batch_size, train_split, False, classifier_type, sequence_len, sequence_stride, shuffle=True)


iter_num = 0
best_val_loss = 1e9
iter_num_on_load = 0
model = optimizer = scaler = None


def load_model():
    global model, optimizer, scaler, iter_num, best_val_loss, iter_num_on_load
    if sequence_len > 1:
        dino_model = Dinov2ForTimeSeriesClassification
    else:
        dino_model = Dinov2ForClassification
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        model = dino_model.from_pretrained(
            f"facebook/dinov2-{dino_size}", id2label=id2label, num_labels=len(id2label), classifier_type=classifier_type)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, checkpoint_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = dino_model.from_pretrained(
            f"facebook/dinov2-{dino_size}", id2label=id2label, num_labels=len(id2label), classifier_type=classifier_type)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        iter_num_on_load = iter_num
        best_val_loss = checkpoint['best_val_loss']
        del state_dict  # very important to clear this checkpoint reference

    for module in model.modules():
        module.train()

    # freeze or unfreeze model
    for name, param in model.named_parameters():
        if name.startswith("dinov2"):
            param.requires_grad = fine_tune
        # maybe also freeze layernorm no matter what here

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(e)
            print("Probably trying to fine-tune full model after resuming from only last layer tuned model, proceeding with reset iter num for learning rate warmup.")
            iter_num = 0
        # iter_num = 0
        # Issue when loading model VRAM usage is higher, therefore OOMs with same params
        # https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213/3
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)    # requires PyTorch 2.0

    return model


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        dataloader_iter = iter(train_dataloader if split == 'train' else val_dataloader)
        losses = torch.zeros(eval_iters * gradient_accumulation_steps)
        for k in range(eval_iters * gradient_accumulation_steps):
            X, Y = get_batch(dataloader_iter)
            with ctx:
                logits, loss = model(X, labels=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def calc_accuracy(logits, labels):
    # Convert logits and labels to numpy arrays
    preds = logits.detach().cpu()
    if classifier_type == "bce":
        preds = torch.nn.functional.sigmoid(preds)
        preds = (preds.numpy() >= 0.5)
        labels = labels.detach().cpu().numpy().astype(bool)
        if sequence_len > 1:
            labels = labels[:, -1, :]   # get the last label in the sequence
        return np.mean(np.all(preds == labels, axis=-1))
    else:
        preds = torch.argmax(preds, dim=-1).numpy()
        labels = labels.detach().cpu().numpy()
        if sequence_len > 1:
            labels = labels[:, -1, :]
        labels = np.argmax(labels, axis=-1)
        return np.mean(preds == labels)





def plot_metrics(metrics, window_size=50):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss and validation loss
    ax1.plot(range(len(metrics["losses"])), metrics["losses"], label='Training Loss', color='blue', alpha=0.3)
    smoothed_losses = moving_average(metrics["losses"], window_size)
    half_window = window_size // 2
    ax1.plot(range(half_window-1, len(metrics["losses"]) - half_window), smoothed_losses, label='Smoothed Training Loss', color='blue')

    ax1.plot(metrics["val_loss_iters"], metrics["val_losses"], label='Validation Loss', color='red', linestyle='dashed', marker='o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax1.set_title('Training Metrics')

    ax2 = ax1.twinx()
    ax2.plot(range(len(metrics["accuracy"])), metrics["accuracy"], label='Training Accuracy', color='green',
             alpha=0.3)
    smoothed_accuracies = moving_average(metrics["accuracy"], window_size)
    ax2.plot(range(half_window - 1, len(metrics["accuracy"]) - half_window), smoothed_accuracies,
             label='Smoothed Training Accuracy', color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, metrics_name))
    plt.close(fig)


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(dataloader_iter):
    x, y = next(dataloader_iter)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def train_loop():
    global iter_num, best_val_loss
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    metrics_dict = {"losses": [], "val_losses": [], "val_loss_iters": [], "accuracy": []}
    model = load_model()
    dataloader_iter = iter(train_dataloader)
    X, Y = get_batch(dataloader_iter)
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and iter_num != iter_num_on_load and iter_num != 0:
            losses = estimate_loss(model)
            metrics_dict["val_losses"].append(losses["val"])
            metrics_dict["val_loss_iters"].append(local_iter_num)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val'] if losses['val'] < best_val_loss else best_val_loss
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config_dict,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))
                    if best_val_loss == losses['val']:
                        torch.save(checkpoint, os.path.join(out_dir, checkpoint_name.split(".")[0] + "-best." + checkpoint_name.split(".")[1]))
            plot_metrics(metrics_dict)
        if iter_num == 0 and eval_only:
            break

        lossf = 0
        accuracy = 0
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, labels=Y)
                loss = loss / gradient_accumulation_steps   # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(dataloader_iter)
            lossf += loss.item()
            accuracy += calc_accuracy(logits, Y)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        accuracy = accuracy / gradient_accumulation_steps
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        metrics_dict["losses"].append(lossf)
        metrics_dict["accuracy"].append(accuracy)
        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            # lossf = loss.item() * gradient_accumulation_steps

            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, accuracy {accuracy*100:.3f}%, lr {lr:.6f}")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break


def gen_data_from_dict_list(dict_list, incorporate_fps, shuffle=True):
    res_list = []
    for filename_dict in dict_list:
        res_list += [get_sequenced_data(filename_dict, incorporate_fps, shuffle)]
    return res_list


def get_sequenced_data(filename_dict, incorporate_fps, shuffle=True):
    # don't concat to np arr cuz first need to seq
    images, labels = utils.concat_data_from_dict(filename_dict, concat=False)
    combined_seq_data = None
    for i in range(len(labels)):
        images_curr, labels_curr = utils.convert_labels_to_time_pressed(labels[i], images=images[i])
        curr_seq_data = generate_timeseries(images_curr, labels_curr, shuffle=shuffle, incorporate_fps=incorporate_fps)
        if not combined_seq_data:
            combined_seq_data = curr_seq_data
        else:
            combined_seq_data = combined_seq_data.concatenate(curr_seq_data)
    return combined_seq_data


def generate_timeseries(images, labels, shuffle=False, incorporate_fps=True):
    sampling_rate = utils.get_fps_ratio() if incorporate_fps else 1
    # labels have to correspond to predicted sequence, not samplerate-1 because we want the next timestep as label
    labels = labels[sampling_rate * config.sequence_len:]
    sequenced_data = timeseries_dataset_from_array(images, labels, sequence_length=config.sequence_len,
                                                   sampling_rate=sampling_rate,
                                                   sequence_stride=config.sequence_stride,
                                                   batch_size=config.BATCH_SIZE,
                                                   shuffle=shuffle)
    return sequenced_data


def test_generate_time_series():
    images, labels = utils.load_file(config.stuck_data_dir_name + config.data_name + "_0.h5")
    print(len(labels))
    data = generate_timeseries(images, labels, shuffle=False, incorporate_fps=True)
    shift_val = utils.get_fps_ratio() * config.sequence_len
    print(len(labels) - shift_val)
    fps_ratio = utils.get_fps_ratio()
    for i, batch in enumerate(data):
        ninputs, nlabels = batch
        print(len(nlabels), len(ninputs))
        for j in range(len(nlabels)):
            temp_l = labels[j + shift_val + i * config.BATCH_SIZE]
            temp_i = images[i * config.BATCH_SIZE + j: i * config.BATCH_SIZE + shift_val + j: fps_ratio]
            print(i * config.BATCH_SIZE + j)
            assert np.array_equal(ninputs[j], temp_i)
            assert np.array_equal(nlabels[j], temp_l)


if __name__ == "__main__":
    train_loop()
