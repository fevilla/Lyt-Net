# LYT-Net


## Experiment

### 1. Create Environment
- Make Conda Environment
```bash
conda create -n LYTNet python=3.10
conda activate LYTNet
```
- Dependencias
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
pip install tensorflow==2.10 opencv-python numpy tqdm matplotlib lpips
```

### 2. Base de datos

LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

LOLv2 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)


<details>

  ```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
  ```

</details>

### 3. Test
[Google Drive](https://drive.google.com/drive/folders/1LgLUXGy-7fQXVnxyEeyBolkZ5ZX1f_em?usp=sharing). 

```bash
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOLv1.h5
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOLv1.h5 --gtmean

python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOLv2_Real.h5
python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOLv2_Real.h5 --gtmean

python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOLv2_Synthetic.h5
python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOLv2_Synthetic.h5 --gtmean
```

### 4. Complejidad Computacional
```bash
python main.py --complexity
python main.py --complexity --shape '(H,W,C)'
```

### 5. Train

```bash
python main.py --train --dataset LOLv1
python main.py --train --dataset LOLv2_Real
python main.py --train --dataset LOLv2_Synthetic
