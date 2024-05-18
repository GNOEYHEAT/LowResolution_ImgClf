# LowResolution_ImgClf
저해상도 조류 이미지 분류 AI 경진대회, DACON (2024.04.08 ~ 2024.05.06)

## Setup
    git clone https://github.com/GNOEYHEAT/LowResolution_ImgClf.git
    cd LowResolution_ImgClf
    conda create -n bird_cls python=3.10 
    conda activate bird_cls
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install -r requirements.txt

- Tested on NVIDIA RTX 3090, A100

## Dataset
- The dataset can be downloaded from the [dacon link](https://dacon.io/competitions/official/236251/data).
- Place the downloaded data inside `lowResolution_ImgClf` directory,
<pre><code>
LowResolution_ImgClf
├── data
│   ├── train
│   │   ├── TRAIN_00000.jpg
│   │   ├── ...
│   ├── test
│   │   ├── TEST_00000.jpg
│   │   ├── ...
│   ├── upscale_train
│   │   ├── TRAIN_00000.jpg
│   │   ├── ...
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv


</code></pre>



 or modify `train_csv_path` and `test_csv_path` in `configs/base.yaml`. It is recommended to input the <b>absolute paths</b> for `train_csv_path` and `test_csv_path`.



## Train and Make Submission File
- If you want to train the model and generate the submission file all at once, execute the script below.

```
python main.py
```

### Train
- If you only want to train, execute the script below.
```
python scripts/train.py
```

### Make Submission File
- If you want to generate the submission file using prediction files, execute the script below. 
```
python scripts/submission.py
```
