import os, torch, shutil
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
import random
import matplotlib.pyplot as plt
import timm
import torchmetrics
from tqdm import tqdm
import cv2

# Check if a GPU is available and use it if possible; otherwise, use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set a random seed for reproducibility
torch.manual_seed(2024)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root, transformations=None, im_files=[".jpg", ".jpeg", ".png"]):
        self.transformations = transformations
        # Collect all image paths matching the specified extensions
        self.im_paths = sorted([f for ext in im_files for f in glob(f"{root}/*/*{ext}")])
        
        # Print a warning message if no images are found
        if len(self.im_paths) == 0:
            print(f"No images found in the dataset directory: {root}")
        
        # Initialize dictionaries to store class names and counts
        self.cls_names, self.cls_counts, count = {}, {}, 0
        for im_path in self.im_paths:
            class_name = self.get_class(im_path)
            # If a new class is found, add its name and index
            if class_name not in self.cls_names: 
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else: 
                self.cls_counts[class_name] += 1

    # Extract the class name from the image path
    def get_class(self, path): 
        return os.path.dirname(path).split("/")[-1]
    
    # Return the length of the dataset
    def __len__(self):
        return len(self.im_paths)
    
    # Return the image and corresponding label at a given index
    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.cls_names[self.get_class(im_path)]

        # Apply transformations if defined
        if self.transformations is not None: 
            im = self.transformations(im)
        return im, gt

# Function to create data loaders
def get_dls(root, transformations, bs, split=[0.9, 0.05, 0.05], ns=4):
    ds = CustomDataset(root=root, transformations=transformations)
    total_len = len(ds)
    # Calculate the size of the training, validation, and test datasets
    tr_len = int(total_len * split[0])
    vl_len = int(total_len * split[1])
    ts_len = total_len - (tr_len + vl_len)
    
    # Split the dataset into training, validation, and test sets
    tr_ds, vl_ds, ts_ds = random_split(dataset=ds, lengths=[tr_len, vl_len, ts_len])
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=ns)
    val_dl = DataLoader(vl_ds, batch_size=bs, shuffle=False, num_workers=ns)
    ts_dl = DataLoader(ts_ds, batch_size=1, shuffle=False, num_workers=ns)

    # Return data loaders and class names
    return tr_dl, val_dl, ts_dl, ds.cls_names

# Function for data analysis and class distribution visualization
def data_analysis(root, transformations, save_path=None):
    ds = CustomDataset(root=root, transformations=transformations)
    cls_counts = ds.cls_counts
    cls_names = list(cls_counts.keys())
    counts = list(cls_counts.values())

    fig, ax = plt.subplots(figsize=(20, 10))
    indices = np.arange(len(counts))

    ax.set_xticks(indices)
    ax.set_xticklabels(cls_names, rotation=60)

    # Display data counts for each class as a bar chart
    ax.bar(indices, counts, width=0.7, color="firebrick")
    ax.set_xlabel("Class Names", color="red")
    ax.set_ylabel("Data Counts", color="red")
    ax.set_title(f"Dataset Class Imbalance Analysis")

    # Display count values above each bar
    for i, v in enumerate(counts): 
        ax.text(i - 0.05, v + 2, str(v), color="royalblue")

    plt.tight_layout()

    # Save the plot to a file if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

# Function to convert tensors back to image
def tensor_2_im(t, t_type="rgb"):
    # Define inverse transformations to convert normalized tensor back to image
    gray_tfs = T.Compose([T.Normalize(mean=[0.], std=[1/0.5]), T.Normalize(mean=[-0.5], std=[1])])
    rgb_tfs = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                         T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs

    # Handle grayscale image conversion
    if t_type == "gray":
        return (invTrans(t) * 255).detach().squeeze().cpu().numpy().astype(np.uint8)
    else:
        # Handle RGB image conversion
        return (invTrans(t) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

# Function to visualize a set of images
def visualize(data, n_ims, rows, cmap=None, cls_names=None):
    assert cmap in ["rgb", "gray"], "Please specify whether the image is grayscale or rgb!"

    plt.figure(figsize=(20, 10))
    # Randomly select image indices
    indices = [random.randint(0, len(data) - 1) for _ in range(n_ims)]

    for idx, ind in enumerate(indices):
        im, gt = data[ind]

        # Plot the images
        plt.subplot(rows, n_ims // rows, idx + 1)

        if cmap == "rgb":
            plt.imshow(tensor_2_im(im, "rgb"))
        elif cmap == "gray":
            plt.imshow(tensor_2_im(im, "gray"), cmap="gray")

        plt.axis('off')

        # Display the ground truth label in the title
        if cls_names is not None:
            plt.title(f"GT -> {cls_names[int(gt)]}")
        else:
            plt.title(f"GT -> {gt}")

    plt.tight_layout()
    plt.show()
    plt.close()

# Function to set up the model for training
def train_setup(m):
    return m.to(device).eval(), 100, device, torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=m.parameters(), lr=3e-4)

# Function to move a batch to the specified devices (GPU or CPU)
def to_device(batch, device):
    return batch[0].to(device), batch[1].to(device)

# Metric calculation function
def get_metrics(model, ims, gts, loss_fn, epoch_loss, epoch_acc, epoch_f1):
    preds = model(ims)
    loss = loss_fn(preds, gts)
    pred_labels = torch.argmax(preds, dim=1)
    # Calculate accuracy and update the F1 score
    epoch_acc += (pred_labels == gts).sum().item()
    epoch_f1 += f1_score(pred_labels, gts)

    return loss, epoch_loss + loss.item(), epoch_acc, epoch_f1

# Visualize learning curves
class PlotLearningCurves:
    def __init__(self, tr_losses, val_losses, tr_accs, val_accs, tr_f1s, val_f1s):
        self.tr_losses = tr_losses
        self.val_losses = val_losses
        self.tr_accs = tr_accs
        self.val_accs = val_accs
        self.tr_f1s = tr_f1s
        self.val_f1s = val_f1s
    
    # Function to plot two arrays with labels and colors
    def plot(self, array_1, array_2, label_1, label_2, color_1, color_2):
        plt.plot(array_1, label=label_1, c=color_1)
        plt.plot(array_2, label=label_2, c=color_2)
    
    # Create a new figure for plotting
    def create_figure(self):
        plt.figure(figsize=(10, 5))
    
    # Add labels and display the plot
    def decorate(self, ylabel, xlabel="Epochs"):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    
    # Visualize training and validation losses, accuracy, and F1 scores
    def visualize(self):
        self.create_figure()
        self.plot(self.tr_losses, self.val_losses, "Train Loss", "Validation Loss", "tab:blue", "tab:red")
        self.decorate("Loss Values")
        self.create_figure()
        self.plot(self.tr_accs, self.val_accs, "Train Accuracy", "Validation Accuracy", "tab:blue", "tab:red")
        self.decorate("Accuracy Scores")
        self.create_figure()
        self.plot(self.tr_f1s, self.val_f1s, "Train F1 Score", "Validation F1 Score", "tab:blue", "tab:red")
        self.decorate("F1 Scores")

# Class to save features for CAM
class SaveFeatures():
    features = None

    def __init__(self, m):
        # Register a forward hook to capture the ouput of a layer
        self.hook = m.register_forward_hook(self.hook_fn)
    
    # Function to save the output to capture the output of a layer
    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()
    
    # Remove the hook when done to avoid memory leaks
    def remove(self): self.hook.remove()

# Function to generate a Class Activation Map(CAM)
def getCAM(conv_fs, linear_weights, class_idx):
    bs, chs, h, w = conv_fs.shape
    # calculate the weighted sum of feature maps for the target class
    cam = linear_weights[class_idx].dot(conv_fs[0, :, :,].reshape((chs, h * w)))
    cam = cam.reshape(h, w)
    # Normalize the CAM for better visualization
    return (cam - np.min(cam)) / np.max(cam)

# Inference function for generating and visualization CAMs
def inference(model, device, test_dl, num_ims, row, final_conv, fc_params, cls_names=None):
    # Extract weights from the final fully connected layer
    weight, acc = np.squeeze(fc_params[0].cpu().data.numpy()), 0
    # Create a SaveFeatures object to capture activations from the final convolutional layers
    activated_features = SaveFeatures(final_conv)
    preds, images, lbls = [], [], []

    # Iterate through the test dataset
    for idx, batch in tqdm(enumerate(test_dl)):
        im, gt = to_device(batch, device)
        pred_class = torch.argmax(model(im), dim=1)
        acc += (pred_class == gt).sum().item()
        images.append(im)
        preds.append(pred_class.item())
        lbls.append(gt.item())
    
    # Print the model's accuracy on the test data
    print(f"Accuracy of the model on the test data -> {(acc / len(test_dl.dataset)):.3f}")

    # Set up the figure for displaying the images and their corresponding CAMs
    plt.figure(figsize=(20, 10))
    indices = [random.randint(0, len(images) - 1) for _ in range(num_ims)]

    # Display each selected image with its CAM overlay
    for idx, index in enumerate(indices):
        im = images[index].squeeze()
        pred_idx = preds[index]
        heatmap = getCAM(activated_features.features, weight, pred_idx)

        # Plot the image and overlay the CAM
        plt.subplot(row, num_ims // row, idx + 1)
        plt.imshow(tensor_2_im(im), cmap="gray")
        plt.imshow(cv2.resize(heatmap, (im_size, im_size), interpolation=cv2.INTER_LINEAR), alpha=0.4, cmap='jet')
        plt.axis("off")

        # Display ground truth and predicted class in the title
        if cls_names is not None: 
            plt.title(
                f"GT -> {cls_names[int(lbls[index])]} ; PRED -> {cls_names[int(preds[index])]}",
                color=("green" if cls_names[int(lbls[index])] == cls_names[int(preds[index])] else "red")
            )
        else: 
            plt.title(f"GT -> {lbls[index]} ; PRED -> {preds[index]}")
    
    plt.tight_layout()
    plt.show()

# Define the dataset path and transformation
root = "C:/CODE/defect_detection/data5"
mean, std, im_size = [0.485, 0.456, 0.406], [0.299, 0.224, 0.225], 224
tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

# Output the length of the dataset for debugging purpose
ds = CustomDataset(root=root, transformations=tfs)
print(f"Number of images found: {len(ds)}")

# Proceed only if there are images in the dataset
if len(ds) > 0:
    tr_dl, val_dl, ts_dl, classes = get_dls(root=root, transformations=tfs, bs=16)
    print(len(tr_dl))
    print(len(val_dl))
    print(len(ts_dl))
    print(classes)

# Perform dataset analysis
# data_analysis(root=root, transformations=tfs)

f1_score = torchmetrics.F1Score(task = "multiclass", num_classes = len(classes)).to(device)

# Test the visualization function
# visualize(tr_dl.dataset, 20, 4, "rgb", list(classes.keys()))
# visualize(ts_dl.dataset, 20, 4, "rgb", list(classes.keys()))

# Define and create the model using a pre-trained model
m = timm.create_model("rexnet_150", pretrained=True, num_classes=len(classes))

# Setup the model for training
m, epochs, device, loss_fn, optimizer = train_setup(m)

print(f"Training on {device}")
print("Start training...")

############### Main training logic #####################
if __name__ == '__main__':
    # Directory to save model  
    save_prefix = "model"
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Best values for early stopping
    best_loss, threshold, patience = float('inf'), 0.01, 5
    not_improved = 0

     # Initialize tracking lists before the training loop
    tr_losses, val_losses, tr_accs, val_accs, tr_f1s, val_f1s = [], [], [], [], [], []

    # Training loop
    for epoch in range(epochs):
        m.train()  # Set model to training mode
        epoch_loss, epoch_acc, epoch_f1 = 0, 0, 0
        for idx, batch in tqdm(enumerate(tr_dl), total=len(tr_dl)):
            ims, gts = to_device(batch, device)
            loss, epoch_loss, epoch_acc, epoch_f1 = get_metrics(m, ims, gts, loss_fn, epoch_loss, epoch_acc, epoch_f1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Track training statistics
        tr_loss_to_track = epoch_loss / len(tr_dl)
        tr_acc_to_track = epoch_acc / len(tr_dl.dataset)
        tr_f1_to_track = epoch_f1 / len(tr_dl)

        tr_losses.append(tr_loss_to_track)
        tr_accs.append(tr_acc_to_track)
        tr_f1s.append(tr_f1_to_track)

        print(f"{epoch + 1}-epoch train loss -> {tr_loss_to_track:.3f}, accuracy -> {tr_acc_to_track:.3f}, f1-score -> {tr_f1_to_track:.3f}")

        # Validation phase
        m.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_epoch_loss, val_epoch_acc, val_epoch_f1 = 0, 0, 0
            for idx, batch in enumerate(val_dl):
                ims, gts = to_device(batch, device)
                loss, val_epoch_loss, val_epoch_acc, val_epoch_f1 = get_metrics(m, ims, gts, loss_fn, val_epoch_loss, val_epoch_acc, val_epoch_f1)

            val_loss_to_track = val_epoch_loss / len(val_dl)
            val_acc_to_track = val_epoch_acc / len(val_dl.dataset)
            val_f1_to_track = val_epoch_f1 / len(val_dl)

            val_losses.append(val_loss_to_track)
            val_accs.append(val_acc_to_track)
            val_f1s.append(val_f1_to_track)

            print(f"{epoch + 1}-epoch validation loss -> {val_loss_to_track:.3f}, accuracy -> {val_acc_to_track:.3f}, f1-score -> {val_f1_to_track:.3f}")

            # # Early stopping logic
            # if val_loss_to_track < (best_loss + threshold):
            #     best_loss = val_loss_to_track
            #     torch.save(m.state_dict(), f"{save_dir}/{save_prefix}_best_model100.pth")
            #     not_improved = 0
            # else:
            #     not_improved += 1
            #     if not_improved == patience:
            #         print(f"Stopping early after {patience} epochs without improvement.")
            #         break
            torch.save(m.state_dict(), f"{save_dir}/{save_prefix}_epoch{epoch + 1}.pth")
    
    PlotLearningCurves(tr_losses, val_losses, tr_accs, val_accs, tr_f1s, val_f1s).visualize() 
           
    print("Training completed.")