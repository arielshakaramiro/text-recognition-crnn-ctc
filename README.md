# Text Recognition with CRNN + CTC Loss

A PyTorch implementation of a CRNN (Convolutional Recurrent Neural Network) trained with CTC (Connectionist Temporal Classification) loss for sequence text recognition, demonstrated on a synthetic multi-digit MNIST sequence dataset.

## Overview

Text recognition (OCR) needs to read a *sequence* of characters from an image, not just classify a single character. This repo shows the classic architecture for that:

- **CNN** — extracts visual features (edges, strokes, shapes) from the input image
- **Bidirectional LSTM** — reads the CNN feature sequence left-to-right and right-to-left to model character order and context
- **CTC Loss** — aligns the predicted sequence to the target label without needing per-character segmentation, and collapses repeated/blank predictions into the final string

## Dataset

Since a ready-made "multi-digit sequence" dataset wasn't used, a synthetic one is built on the fly from MNIST: 5 random MNIST digit images are concatenated horizontally into a single `[1, 28, 140]` image per sample, with the digit sequence as the label (e.g. image → label `"31841"`).

## Architecture

```
Input [B, 1, 28, 140]
  → Conv2d(1, 64) + ReLU + MaxPool
  → Conv2d(64, 128) + ReLU + MaxPool
  → reshape to sequence [B, W, C*H]
  → BiLSTM(hidden=128, bidirectional=True)
  → Linear(256, 11)   # 10 digits + 1 CTC blank
  → log_softmax
```

Trained with `nn.CTCLoss(blank=10)` and the Adam optimizer (lr=1e-3) for 10 epochs on Google Colab (GPU). The trained weights are saved to Google Drive (mounted at runtime) rather than only to the ephemeral Colab disk.

## Results (verified — actual Colab run)

Held-out test set: 2,000 sequences generated from MNIST's own test split — images the model never saw during training.

| Epoch | Train Loss | Test Loss (held-out) |
|-------|-----------|----------------------|
| 1     | 2.0011    | 0.5838                |
| 2     | 0.3011    | 0.1850                |
| 3     | 0.1380    | 0.1102                |
| 4     | 0.0867    | 0.0765                |
| 5     | 0.0599    | 0.0732                |
| 6     | 0.0431    | 0.0590                |
| 7     | 0.0308    | 0.0472                |
| 8     | 0.0242    | 0.0432                |
| 9     | 0.0171    | 0.0409                |
| 10    | 0.0138    | 0.0437                |

![Training loss curve](images/training_loss.png)

**Quantitative evaluation**, computed over all 2,000 held-out test sequences:

| Metric | Value |
|---|---|
| Sequence accuracy (exact match) | 93.45% |
| Character Error Rate (CER) | 1.37% |

**Qualitative check** — a random sample from the held-out test set, greedy-decoded:

```
Target   : 94653
Prediksi : 94653
```

![Sample prediction](images/sample_prediction.png)

> **Note on scope:** test loss ticks up slightly between epoch 9 (0.0409) and epoch 10 (0.0437) while train loss keeps falling — a mild early sign of overfitting. The accuracy/CER numbers above still come from the final (epoch 10) weights; a shorter run or early stopping might land on a marginally better checkpoint, but that wasn't explored further here.

## Setup notes

- **Model persistence:** the notebook mounts Google Drive and saves/loads the trained model (`crnn_mnist.pt`) at `/content/drive/MyDrive/text-recognition-crnn-ctc/`, so it survives runtime resets.

## How to run

1. Open `notebooks/text_recognition_crnn_ctc.ipynb` in Google Colab
2. Set the runtime to GPU
3. Run all cells — MNIST downloads automatically on first run, and you'll be prompted for Drive access when the model is saved

## Repo structure

```
.
├── notebooks/
│   └── text_recognition_crnn_ctc.ipynb
├── images/
│   ├── training_loss.png
│   └── sample_prediction.png
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
