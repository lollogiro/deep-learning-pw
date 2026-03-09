# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.13.11)
#     language: python
#     name: python3
# ---

# %%
# TO CREATE A CORRESPONDENT .py
# jupytext --set-formats ipynb,py:percent test.ipynb

# %%
import pickle

N_FOLDS = 6
N_INNER_MODELS = 5

MODEL_NAME = "mobilenetv3"

history_path = f'{MODEL_NAME}_train_history.pkl'

with open(history_path, 'rb') as f:
    history = pickle.load(f)

val_losses = history['val_losses']
best_checkpoints = []

for test_idx in range(N_FOLDS):
    test_fold = test_idx + 1
    
    available_val_folds = [f for f in range(1, N_FOLDS + 1) if f != test_fold]
    
    start_idx = test_idx * N_INNER_MODELS
    end_idx = start_idx + N_INNER_MODELS
    current_fold_losses = val_losses[start_idx:end_idx]
    
    best_loss = float('inf')
    best_inner_idx = -1
    
    for i, losses in enumerate(current_fold_losses):
        min_epoch_loss = min(losses)
        if min_epoch_loss < best_loss:
            best_loss = min_epoch_loss
            best_inner_idx = i
            
    best_val_fold = available_val_folds[best_inner_idx]
    
    model_filename = f'test_{test_fold}_val_{best_val_fold}_best.pt'
    best_checkpoints.append(model_filename)
    
    print(f"Test Fold {test_fold} | Best Validation Fold: {best_val_fold} | Validation Loss: {best_loss:.4f} -> {model_filename}")

BEST_MODELS_PATHS = []
for checkpoint in best_checkpoints:
    BEST_MODELS_PATHS.append(f'./{MODEL_NAME}_results/{checkpoint}')

print("\nBest Model Checkpoints:")
for path in BEST_MODELS_PATHS:
    print(path)

# %%
import torch
import pandas as pd
import timm

CLASS_NAMES = [
    "overcast",
    "sunshine",
    "snow",
    "hail",
]

def load_best_model_eval(model_card, num_classes, checkpoint_path, device):
    model = timm.create_model(model_card, pretrained=False, num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

def run_ensemble_inference(model_paths, dataloader, model_card, device):
    print(f"Starting ensemble inference with {len(model_paths)} models...")
    
    fold_ensemble_logits = []
    
    for i, model_path in enumerate(model_paths):
        print(f"Loading and predicting with model {i+1}/{len(model_paths)}: {model_path}")
        
        model = load_best_model_eval(model_card, len(CLASS_NAMES), model_path, device)
        model.eval()
        
        all_batch_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(device)
                
                outputs = model(inputs)
                all_batch_logits.append(outputs.cpu())
                
        model_logits = torch.cat(all_batch_logits)
        fold_ensemble_logits.append(model_logits)
        
        del model
        torch.cuda.empty_cache()

    ensemble_logits_stack = torch.stack(fold_ensemble_logits)
    avg_logits = torch.mean(ensemble_logits_stack, dim=0)
    
    predicted_indices = torch.argmax(avg_logits, dim=1).numpy()
    predicted_classes = [CLASS_NAMES[idx] for idx in predicted_indices]
    
    return avg_logits, predicted_classes


# %%
import cv2
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm

MNV3_CARD = 'timm/mobilenetv3_small_100.lamb_in1k'

# MNV4_CARD = 'timm/mobilenetv4_conv_small.e2400_r224_in1k'

N_FOLDS = 6
N_INNER_MODELS = 5

DATA_CONFIG = timm.data.resolve_data_config({}, model=MNV3_CARD)

IMG_SIZE = DATA_CONFIG['input_size'][1]

TRANSFORMS_VAL = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=DATA_CONFIG['mean'], std=DATA_CONFIG['std'])
])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_video_ensemble(video_path, model_paths, model_card, device, batch_size=32, frame_skip=1):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nLoading ensemble models for {video_path.name}...")
    models = []
    for path in model_paths:
        model = load_best_model_eval(model_card, len(CLASS_NAMES), path, device)
        model.eval()
        models.append(model)

    results = []
    batch_frames = []
    batch_frame_indices = []
    
    current_frame = 0
    
    print(f"Processing {total_frames} frames (evaluating 1 every {frame_skip} frames)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if len(batch_frames) > 0:
                batch_tensor = torch.stack(batch_frames).to(device)
                
                fold_logits = []
                with torch.no_grad():
                    for model in models:
                        outputs = model(batch_tensor)
                        fold_logits.append(outputs.cpu())
                
                ensemble_logits_stack = torch.stack(fold_logits)
                avg_logits = torch.mean(ensemble_logits_stack, dim=0)
                
                predicted_indices = torch.argmax(avg_logits, dim=1).numpy()
                predicted_classes = [CLASS_NAMES[idx] for idx in predicted_indices]
                
                for idx, pred_class in zip(batch_frame_indices, predicted_classes):
                    results.append({
                        'video_path': str(video_path),
                        'frame_number': idx,
                        'weather_prediction': pred_class
                    })
            break
            
        if current_frame % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor_img = TRANSFORMS_VAL(pil_img)
            batch_frames.append(tensor_img)
            batch_frame_indices.append(current_frame)
            
        if len(batch_frames) == batch_size:
            batch_tensor = torch.stack(batch_frames).to(device)
            
            fold_logits = []
            with torch.no_grad():
                for model in models:
                    outputs = model(batch_tensor)
                    fold_logits.append(outputs.cpu())
            
            ensemble_logits_stack = torch.stack(fold_logits)
            avg_logits = torch.mean(ensemble_logits_stack, dim=0)
            
            predicted_indices = torch.argmax(avg_logits, dim=1).numpy()
            predicted_classes = [CLASS_NAMES[idx] for idx in predicted_indices]
            
            for idx, pred_class in zip(batch_frame_indices, predicted_classes):
                results.append({
                    'video_path': str(video_path),
                    'frame_number': idx,
                    'weather_prediction': pred_class
                })
                
            batch_frames = []
            batch_frame_indices = []

        current_frame += 1

    cap.release()
    
    for model in models:
        del model
    torch.cuda.empty_cache()
    
    return pd.DataFrame(results)


def run_all_videos(root_dir, model_paths, model_card, device):
    root_path = Path(root_dir)
    
    video_files = list(root_path.rglob('*.mp4'))
    print(f"Found {len(video_files)} videos to process.")
    
    for i, video_path in enumerate(video_files):

        # if i >= 1:
        #     print(f"DEBUG: Processed {i} videos.")
        #     break

        csv_path = video_path.with_suffix('.csv')
        
        if csv_path.exists():
            print(f"Skipping {video_path.name}, already processed.")
            continue
            
        print(f"Processing {video_path} ({i+1}/{len(video_files)})")
        
        df_results = process_video_ensemble(video_path, model_paths, model_card, device, frame_skip=1, batch_size=32)
        
        df_results.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

def reset_output(root_dir):
    root_path = Path(root_dir)
    video_files = list(root_path.rglob('*.mp4'))
    
    for video_path in video_files:
        csv_path = video_path.with_suffix('.csv')
        if csv_path.exists():
            print(f"Removing existing CSV: {csv_path}")
            csv_path.unlink()

# reset_output('/home/lorenzo/data/Videos/') # to clean the folder from previous inference results

run_all_videos('/home/lorenzo/data/Videos/', BEST_MODELS_PATHS, MNV3_CARD, DEVICE)
