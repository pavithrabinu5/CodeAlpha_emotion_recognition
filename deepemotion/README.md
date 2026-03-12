# DeepEmotion — Speech Emotion Recognition

End-to-end PyTorch pipeline for classifying emotions from speech (happy, sad, angry, neutral, fear, disgust, surprise) with reproducible preprocessing, training, evaluation, and a Gradio demo.

## Project layout
```
deepemotion/
├── app/               # Gradio demo
├── data/              # Place raw datasets (ravdess/, tess/, emodb/)
├── models/            # Saved checkpoints
├── notebooks/         # Example notebooks
├── outputs/           # Logs, figures, processed audio, features
├── src/               # Python modules (preprocess, augment, feature extraction, training, inference)
├── utils/             # Shared helpers
├── requirements.txt
└── README.md
```

## Quickstart
1. Create env & install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r deepemotion/requirements.txt
```
2. Download datasets into `deepemotion/data/` with subfolders `ravdess/`, `tess/`, `emodb/` (each containing wav files).
3. Preprocess + (optional) augment
```bash
python deepemotion/src/preprocess.py --data_root deepemotion/data --out_csv deepemotion/outputs/metadata.csv --clean_dir deepemotion/outputs/clean_wav --augment
```
4. Extract features (Mel + MFCC)
```bash
python deepemotion/src/feature_extraction.py --metadata deepemotion/outputs/clean_wav/metadata_clean.csv --out_dir deepemotion/outputs/features --specaug
```
5. Train (choose model: cnn | bilstm | hybrid)
```bash
python deepemotion/src/train.py --metadata deepemotion/outputs/features/metadata_features.csv --model cnn --feature_type mel --epochs 30 --batch_size 32
```
6. Evaluate
```bash
python deepemotion/src/evaluate.py --metadata deepemotion/outputs/features/metadata_features.csv --checkpoint deepemotion/models/best_cnn.pt --feature_type mel --out_dir deepemotion/outputs/eval
```
7. Predict a single file
```bash
python deepemotion/src/predict.py --audio sample.wav --checkpoint deepemotion/models/best_cnn.pt
```
8. Launch Gradio demo
```bash
python deepemotion/app/gradio_app.py --checkpoint deepemotion/models/best_cnn.pt
```

## Notes
- Scripts assume 16 kHz mono audio; preprocessing handles resampling, trimming, light noise reduction, and optional augmentation.
- Class imbalance is mitigated with a weighted sampler during training.
- TensorBoard logs are written to `outputs/logs`.
- Evaluation saves confusion matrix and ROC curves to `outputs/eval`.

## Example notebook
`notebooks/DeepEmotion_demo.ipynb` contains a lightweight walkthrough of preprocessing, feature extraction, and quick inference.

## Reproducibility
Use `seed_everything` (already called in scripts) and keep track of package versions from `requirements.txt`.
