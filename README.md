# Platelet Cell Classification with YOLO11

Image classification of platelet cells under four experimental conditions, using YOLO11 classification models from Ultralytics. This repository is a follow-up to my MSc dissertation at the University of Hull, *Deep Learning Models for the Automatic Classification of Platelet Cells* (December 2023), which compared custom CNNs, VGG16, VGG19 and a checkpoint ensemble on the same images.

## Task

Microscopy images of platelets are classified into four classes:

| Class | Condition |
| --- | --- |
| Control | Untreated platelets |
| Control + Zinc | Platelets with added zinc |
| Milrinone | Platelets treated with milrinone |
| Milrinone + Zinc | Platelets treated with milrinone and zinc |

Zinc and milrinone change the shape platelets take (spreading and its reversal), so the goal is to tell the conditions apart from cell appearance alone. A model that does this reliably could support tools that estimate the level of zinc or other substances from platelet images.

## Data

The images come from the study by Coupland et al. (2023), *Platelet zinc status regulates prostaglandin-induced signaling, altering thrombus formation*, Journal of Thrombosis and Haemostasis. The dataset is not open access, so **no images are included in this repository**.

Preprocessing, as used in the dissertation:

- The original Carl Zeiss (CZI) images were converted to JPG.
- Each large image was tiled into a 5x5 grid of non-overlapping patches, and each patch is treated as an independent input.
- Data was split into training, validation and test sets in a 70/20/10 ratio.
- Two versions of the training data were used: without augmentation (tiling only) and with augmentation (rotation, shifts, horizontal and vertical flips).

[TODO: confirm the YOLO input image size, the number of epochs, and that the augmented set is the same as in the dissertation.]

## Models

Five YOLO11 classification models were trained, from nano to extra-large:

`yolo11n-cls`, `yolo11s-cls`, `yolo11m-cls`, `yolo11l-cls`, `yolo11x-cls`

Each was trained with and without augmentation, and both the `best.pt` and `last.pt` checkpoints were evaluated on the held-out test set.

## Results

All figures are percentages. Precision and recall are macro-averaged over the four classes. The validation and test sets are identical across all runs.

[TODO: confirm the "Val. acc." column is validation accuracy, and confirm the averaging method for precision and recall.]

### Without augmentation

| Model | Checkpoint | Val. acc. | Test acc. | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| YOLO11n | best | 76.4 | 73.9 | 73.9 | 73.9 |
| YOLO11n | last | 76.4 | 77.6 | 77.7 | 77.7 |
| YOLO11s | best | 77.4 | 83.9 | 84.7 | 83.9 |
| YOLO11s | last | 77.4 | 80.7 | 81.5 | 80.8 |
| YOLO11m | best | 79.6 | 82.0 | 82.8 | 82.3 |
| YOLO11m | last | 79.6 | 82.0 | 82.0 | 82.2 |
| YOLO11l | best | 82.2 | 80.1 | 80.4 | 80.1 |
| YOLO11l | last | 82.2 | 80.1 | 80.4 | 80.1 |
| YOLO11x | best | 77.7 | 74.5 | 74.8 | 74.7 |
| YOLO11x | last | 77.7 | 80.1 | 80.3 | 80.3 |

### With augmentation

| Model | Checkpoint | Val. acc. | Test acc. | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| YOLO11n | best | 82.8 | 82.0 | 82.5 | 82.1 |
| YOLO11n | last | 82.8 | 83.2 | 83.8 | 83.4 |
| YOLO11s | best | 84.1 | 82.0 | 82.7 | 82.3 |
| YOLO11s | last | 84.1 | 82.0 | 82.7 | 82.3 |
| YOLO11m | best | 84.1 | 87.6 | 88.4 | 88.0 |
| YOLO11m | last | 84.1 | 90.1 | 90.7 | 90.5 |
| YOLO11l | best | 85.4 | 84.5 | 85.1 | 84.7 |
| YOLO11l | last | 85.4 | 86.3 | 87.2 | 86.6 |
| YOLO11x | best | 83.4 | 85.1 | 86.5 | 85.5 |
| YOLO11x | last | 83.4 | 85.1 | 86.5 | 85.5 |

Full results are in `Mcellsv11.csv`.

### Key findings

- Augmentation helped every model size. With augmentation, test accuracy ranged from 82% to 90%. Without it, the range was 74% to 84%.
- The highest test accuracy was YOLO11m (`last.pt`, with augmentation) at 90.1%. The highest validation accuracy was YOLO11l with augmentation, whose test accuracy was 84.5% (`best.pt`) and 86.3% (`last.pt`). Because the top test score is the best of many runs, the range across models is a fairer summary than any single number.
- Larger models were not consistently better. YOLO11m and YOLO11l did best with augmentation, and the nano and small models were competitive.

### Comparison with the dissertation

In the dissertation, the best result on this four-class task was a checkpoint ensemble of CNN and VGG models, which reached 80% test accuracy, precision and recall with augmentation. The best single model in that ensemble reached 71%. Single models overfitted heavily because each class had few images.

[TODO: only state that the YOLO11 models improve on the dissertation if the test set is the same as the dissertation test set. The dissertation four-class test set has 100 images.]

## Repository contents

| File | Description |
| --- | --- |
| `Platelets - Yolov11.ipynb` | Training and evaluation notebook |
| `Mcellsv11.csv` | Evaluation results for all runs |
| `runs/classify/` | Ultralytics training outputs |
| `yolo11*-cls.pt` | Pretrained YOLO11 classification weights used as starting points |

## How to run

```bash
pip install ultralytics
```

The dataset must be arranged in the folder layout Ultralytics expects for classification, with one subfolder per class inside `train`, `val` and `test`:

```
dataset/
  train/
    Control/
    Control_Zinc/
    Milrinone/
    Milrinone_Zinc/
  val/
  test/
```

Then train and evaluate:

```python
from ultralytics import YOLO

model = YOLO("yolo11m-cls.pt")
model.train(data="path/to/dataset", epochs=..., imgsz=...)  # use the values from the notebook
metrics = model.val(split="test")
```

## Limitations

- The dataset is small. Each class has relatively few images, and tiling means some patches contain very few cells, which can make classes harder to separate.
- The images are not public, so the results cannot be reproduced without access to the original dataset.
- Results come from a single split and a single run per configuration, so small differences between models should not be over-read.

## Next steps

- Add results for YOLOv8 and YOLOv9 on the same images.
- Try YOLO26 on the same images.
- Test other transfer learning models and combine models in an ensemble, as suggested in the dissertation.

## Reference

Coupland, C. A., Naylor-Adamson, L., Booth, Z., Price, T. W., Gil, H. M., Firth, G., Avery, M., Ahmed, Y., Stasiuk, G. J. & Calaminus, S. D. (2023) Platelet zinc status regulates prostaglandin-induced signaling, altering thrombus formation. *Journal of Thrombosis and Haemostasis*.

## Author

Olalekan Oyeleye | [LinkedIn](https://www.linkedin.com/in/olalekanoyeleye) | olalekanoyeleye@yahoo.com
