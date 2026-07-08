# Tactile Slip Phase Learning

Code for tactile image slip-phase labeling and model training.

The project targets three frame-level classes:

- `0`: stable
- `1`: incipient_slip
- `2`: translational_slip

The main workflow is:

```text
flat tactile images
    -> generate frame labels with marker-motion / rigid-residual analysis
    -> train ResNet18 CNN on single frames
    -> extract CNN feature matrix [num_trials, sequence_length, 512]
    -> train LSTM on temporal feature sequences
    -> predict slip phase for new trials
```

## Kaggle Data Format

The scripts assume flat Kaggle image files like:

```text
/kaggle/input/datasets/shishensaiweng/kaggle-flat-90trials-images-and-labels/
  trial_001_repeat_01_center_x_pos_slide_sequence__frame_02.png
  trial_001_repeat_01_center_x_pos_slide_sequence__frame_03.png
  ...
```

The label-generation script can generate labels directly from image names and images.
It does not need an old label CSV.

## Scripts

Run these in order for the full pipeline:

1. `scripts/01_generate_labels_from_flat_images.py`
   - Generates slip labels from flat image files only.
   - Main output:
     `/kaggle/working/generated_rigid_residual_labels_from_images_only/generated_training_labels_excluding_frame01.csv`

2. `scripts/02_train_resnet18_cnn.py`
   - Trains a ResNet18 frame classifier.
   - Input: flat images + frame-level label CSV.
   - Main output:
     `/kaggle/working/best_resnet18_flat_slip_phase.pth`

3. `scripts/03_unsupervised_clustering_baseline.py`
   - Optional unsupervised baseline using motion features + PCA + KMeans/GMM.
   - Useful as a comparison with supervised CNN.

4. `scripts/04_extract_cnn_feature_matrix.py`
   - Loads trained CNN and extracts one 512-dim feature vector per frame.
   - Main outputs:
     `cnn_feature_matrix.npy`, `cnn_label_matrix.npy`, `cnn_feature_trial_info.csv`

5. `scripts/05_train_lstm_from_cnn_features.py`
   - Trains LSTM from precomputed CNN feature matrix.
   - Main output:
     `/kaggle/working/best_lstm_from_cnn_features.pth`

6. `scripts/06_train_end_to_end_cnn_lstm.py`
   - Alternative one-step CNN+LSTM training script.
   - It extracts CNN features during LSTM training instead of using saved `.npy` features.

7. `scripts/07_predict_new_trial_cnn_lstm.py`
   - Predicts slip phases for a new trial using trained CNN + LSTM.

8. `scripts/optional_label_raw_trial_folders.py`
   - Optional script for raw non-flat trial-folder datasets.
   - Use only when images are arranged as `session/trial_xxx/frame_XX.png`.

## Notes

- Kaggle `/kaggle/input` is read-only. Save generated outputs to `/kaggle/working`.
- `frame_01` is ignored when present because it often corresponds to contact/capture settling.
- For GitHub, do not upload large image datasets, `.pth` model weights, `.npy` feature matrices, or generated CSV outputs unless intentionally publishing a small sample.

