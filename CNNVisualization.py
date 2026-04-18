import pickle
import os
from train_cnn import TwoBlockCNN, load_split, get_activation
import torch
import matplotlib.pyplot as plt
import numpy as np

def init_label_codes():
        # Mapping dictionary
    label_codes = {
        0: 'aircraft',
        1: 'car',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    return label_codes

def save_layer_plots(activations, file_path, filters_count, label_codes, true_label, predicted, layer_name):
    if filters_count%4 != 0:
        return "You must select a filters_count multiple of 4"
    # --- Layer feature maps: 32 filters at 32x32 ---
    fig, axes = plt.subplots(int(filters_count/8), 8, figsize=(16, int(filters_count/4)))
    for i, ax in enumerate(axes.flat):
        ax.imshow(activations[layer_name][0, i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'f{i}', fontsize=7)
    plt.suptitle(f'{layer_name.title()} feature maps ({filters_count} filters, {activations[layer_name].shape[2]}x{activations[layer_name].shape[3]})\n'
                f'true: {label_codes[true_label]}, predicted: {label_codes[predicted]}')
    plt.tight_layout()
    fig.savefig(os.path.join(f'{file_path}/out/', f'{layer_name}.png'), dpi=150, bbox_inches='tight')
    plt.show()

def main():
    DATA_DIR = os.path.dirname(__file__)
    label_codes = init_label_codes()
    x_test, y_test = load_split(f'{DATA_DIR}/res/test/', ['test_batch_sigma_0_00'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load model ----
    model = TwoBlockCNN(dropout_rate=0.5).to(device) # recreate architecture without weights
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "two_block_cnn.pth"),
                                    map_location=device)) # add weights (now you have pretrained model)
    model.eval() # evaluation mode ( no weight updates )
    # Re-register hooks
    activations = {}
    model.conv1.register_forward_hook(get_activation('conv1', activations))
    model.pool1.register_forward_hook(get_activation('pool1', activations))
    model.conv2.register_forward_hook(get_activation('conv2', activations))
    model.pool2.register_forward_hook(get_activation('pool2', activations))


    # Pick your test image
    img_idx = 400
    img_tensor = torch.from_numpy(x_test[img_idx:img_idx+1]).to(device)

    # Forward pass on your chosen image — this populates activations
    with torch.no_grad():
        logits = model(img_tensor)
        predicted = logits.argmax(dim=1).item()

    # Now activations correspond to x_test[img_idx]
    # Save if you want
    activations_np = {k: v.cpu().numpy() for k, v in activations.items()}
    with open(os.path.join(DATA_DIR, 'activations.pkl'), 'wb') as f:
        pickle.dump(activations_np, f)

    with open(os.path.join(DATA_DIR, 'activations.pkl'), 'rb') as f:
        activations = pickle.load(f)

    # activations is now a plain dictionary of numpy arrays
    print(activations.keys())          # dict_keys(['conv1', 'conv2'])
    print(activations['conv1'].shape)  # (1, 32, 32, 32)
    print(activations['conv2'].shape)  # (1, 64, 8, 8)

    # Pick a single test image
    img_tensor = torch.from_numpy(x_test[img_idx:img_idx+1]).to(device)  # shape (1, 3, 32, 32)
    true_label = y_test[img_idx]

    with torch.no_grad():
        logits = model(img_tensor)
        predicted = logits.argmax(dim=1).item()

    # activations now contains the feature maps from conv1 and conv2
    print(activations.keys())         # dict_keys(['conv1', 'conv2'])
    print(activations['conv1'].shape) # (1, 32, 32, 32) — 32 filters, 32x32
    print(activations['conv2'].shape) # (1, 64, 8, 8)  — 64 filters, 16x16
    print(type(activations['conv2'].shape[2]))

    # --- Show original image ---
    # x_test is normalized so denormalize for display
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std  = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
    img_display = x_test[img_idx] * std + mean          # denormalize
    img_display = np.clip(img_display, 0, 1)
    img_display = img_display.transpose(1, 2, 0)  # (3,32,32) -> (32,32,3)

    plt.figure(figsize=(3, 3))
    plt.imshow(img_display)
    plt.axis('off')
    plt.title(f'Input — true: {label_codes[true_label]}, '
            f'predicted: {label_codes[predicted]}')
    plt.savefig(os.path.join(f'{DATA_DIR}/out/', 'x_test.png'), dpi=150, bbox_inches='tight')
    plt.show()

    #def save_layer_plots(activations, file_path, filters_count, label_codes, true_label, predicted, layer_name, pixel_dim):
    # Conv1 feature maps: 32 filters at 32x32
    save_layer_plots(activations, DATA_DIR, 32, label_codes, true_label, predicted, "conv1")

    # Pool1 feature maps: 32 features at 16x16
    save_layer_plots(activations, DATA_DIR, 32, label_codes, true_label, predicted, 'pool1')

    # Conv2 feature maps: 64 filters at 16x16
    save_layer_plots(activations, DATA_DIR, 64, label_codes, true_label, predicted, 'conv2')
    # Pool2 feature maps: 64 filters at 8x8
    save_layer_plots(activations, DATA_DIR, 64, label_codes, true_label, predicted, 'pool2')


if __name__ == '__main__':
    main()