# OpenVINO resnet_v1_50 Guide

## Prerequest

### Env & datasets

1. 2018_R5-2019_R1.1 (currently installed in /root/dldt)
2. Preprocessed ImageNet val (224 * 224 * 3) (currently in /home/ubuntu/intel)
3. Export resnet_v1_50.pb, and download ckpb
4. val_bmp_32 and `val.txt`

Note that OpenVINO includes normalization if `mean_values` or `scale` are set. So, during preprocessing we only need `central_crop` and `resize`, then save preprocessed image into `bmp` (not `jpeg` because `jpeg` has lossy compression, which leads to lower accuary).

### Key parameters

* `val.txt` for calibration and validation. **Suggest to put `val.txt` out of val/image dir, to avoid read val.txt as image by mistake.**

```bash
# example of val.txt
image_path predict_label
```

* `--report_type detailed_counters` in benchmark_app for dumping benchmark result. Using `--dump` oin validation_app for dumping detailed predict results.

* `-b` batch size. We can use it to replace `input_shape` with `mo_tf.py`. Note that `-b` will encounter error on OpenVINO 2018 `benchmark_app`.

* `-nireq`: Number of inference requests (threads) in parallel. It should be no larger than core number.

## Prepare OpenVINO model

### Pre-pare resnet_v1_50.pb and resnet_v1_50.ckpb (optional)

```bash
# resnet_v1_50.ckpb
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz

# resnet_v1_50.pb
git clone https://github.com/tensorflow/models.git
python ~/models/research/slim/export_inference_graph.py --alsologtostderr --model_name=resnet_v1_50 --labels_offset=1 --output_file=./resnet_v1_50.pb
```

### Model Optimization

```bash
# Convert TensorFlow model into OpenVINO IR
/root/dldt/model-optimizer/mo_tf.py --input_model resnet_v1_50.pb --input_checkpoint resnet_v1_50.ckpt --input_shape [4,224,224,3] --reverse_input_channels --mean_values [123.68,116.78,103.94]
/root/dldt/inference-engine/bin/intel64/Release/validation_app -m resnet_v1_50.xml -i /mnt/nvm/val_resized/val.txt
```

**Note that Top 1 and Top 5 accuracy of resnet_v1_50 should be close to 75.2% and 92.2%.**

### Calibration/Quantization (Int8 and VNNI)

Calibration requires a small subset of validation dataset, e.g., 32. Herein we use first 32 image with `-subset 32`.

```bash
# Calibrate/quantize optmized model
/root/dldt/inference-engine/bin/intel64/Release/calibration_tool -m resnet_v1_50.xml -i /mnt/nvm/val_resized/val.txt -subset 32
# prepare labels for quantizd model
cp resnet_v1_50.labels resnet_v1_50_i8.labels
# validate quantizd model
/root/dldt/inference-engine/bin/intel64/Release/validation_app -m resnet_v1_50_i8.xml -i /mnt/nvm/val_resized/val.txt
```

## Benchmark

```bash
# Optmized model benchmark in sync mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'sync' -d CPU -m resnet_v1_50.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4
# Optmized model 20 parallel (-nireq 20) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m resnet_v1_50.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4 -nireq 20
# Int8 Quantizd model 56 parallel (-nireq 56) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m resnet_v1_50_i8.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt
/nvm/val_resized/ -b 4 -nireq 56
```
