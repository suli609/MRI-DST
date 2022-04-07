## Requirements

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files. If `ffmpeg` isn't in your system path (ie. `which ffmpeg` doesn't return its path, or you're on an OS other than *nix), you'll need to update the path to `ffmpeg` in `data/2_extract_files.py`.

## 数据
训练和测试数据存放在 ./data/train 和./data/test目录下
数据目录在 ./data/data_file.csv中

## 模型训练
训练命令：
python train.py -g 1 -model_dir ./data/ckpt/test -model_name c3d
    -g 第几张显卡
    -model_dir 训练模型存放地址
    -model_name 保存的模型名称

通常1000个以内的epoch模型即可训练完成

## 模型预测
手动更改validate_rnn.py中的 saved_model
预测命令：
python validate_rnn.py -g 1 -val_steps 30
    -g 第几张显卡
    -val_steps 测试集总共30个样本，此处设置为30




