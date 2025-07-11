# ⚙️ Mechanical-Parts Image Classifier  

A production-ready pipeline that **detects four mechanical parts—bolt, nut, washer and locating-pin—straight from RGB images**.  

* ✅ automated Kaggle download / extraction  
* ✅ clean train/val folder split  
* ✅ transfer-learning training script (EfficientNet-B0)  
* ✅ evaluation + pretty confusion-matrix  
* ✅ one-liner inference script with saved weights  
* ✅ pre-split dataset & pretrained weights, so you can play immediately

---

## 🌳 Project Structure

```text
PROJECT DIRECTORY/
├─ datasets/                        # raw + split images (already included 🎉)
│  ├─ mechanical_parts/             # original Kaggle images
│  └─ mechanical_parts_split/       # train/val sub-folders
├─ effnet_weights_final.h5          # 6 MB of pretrained goodness
├─ import_extract_dataset.py        # optional re-download from Kaggle
├─ split.py                         # 80-20 stratified split
├─ train.py                         # TL training → exports best weights
├─ evaluate.py                      # full metrics & confusion matrix
└─ predict.py                       # single-image inference demo
