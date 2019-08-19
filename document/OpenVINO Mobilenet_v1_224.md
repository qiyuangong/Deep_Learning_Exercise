# OpenVINO Mobilenet_v1 Guide

## Prerequests

### Env & datasets

1. 2018_R5-2019_R1.1 (currently installed in /home/ubuntu/intel/dldt)
2. Resize ImageNet val into val_bmp (224 * 224 * 3) (currently in /home/ubuntu/intel)
3. Download mobilenet_v1_1.0_224_frozen.pb
4. val_bmp_32 and `val.txt`

### Key parameters

* `val.txt` for calibration and validation. **Suggest to put `val.txt` out of val/image dir, to avoid read val.txt as image by mistake.**

```bash
# example of val.txt
image_path predict_label
```

* `--report_type detailed_counters` in benchmark_app for dumping benchmark result. Using `--dump` oin validation_app for dumping detailed predict results.

* `-b` batch size. We can use it to replace `input_shape` with `mo_tf.py`. Note that `-b` will encounter error on OpenVINO 2018 `benchmark_app`.

* `-nireq`: Number of inference requests (threads) in parallel. It should be no larger than core number.

* `-Czb`: required in `validation_app` and `calibration_tool`. "Zero is a background" flag. For models that use 0 as background, such as inception and mobilenet.

## Prepare OpenVINO TensorFlow model

### Pre-pare mobilenet_v1.pb and mobilenet_v1.ckpb (optional)

```bash
# mobilenet_v1.pb
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
```

### Model Optimization (Convert to OpenVINO IR)

```bash
# Convert from TensorFlow to OpenVINO model
/home/ubuntu/intel/dldt/model-optimizer/mo_tf.py --input_model mobilenet_v1_1.0_224_frozen.pb -b 4 --reverse_input_channels --mean_values [127.5,127.5,127.5] --scale 127.5
# Validation acc with smalle dataset to ensure everything is correct
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/validation_app -m mobilenet_v1.xml -i /home/ubuntu/intel/val_bmp_32/val.txt -Czb
```

**Note that Top 1 and Top 5 accuracy of mobilenet_v1 should be close to 71.9% and 91%.**

### Calibration/Quantization (Convert IR to Int8 IR)

```bash
# Calibrate/quantize optmized model
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/calibration_tool -m mobilenet_v1.xml -i /home/ubuntu/intel/val_bmp_32/val.txt -subset 32 -Czb
# prepare labels for quantizd model
cp mobilenet_v1.labels mobilenet_v1_i8.labels
# validate quantizd model
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/validation_app -m mobilenet_v1_i8.xml -i /home/ubuntu/intel/val_bmp_32/val.txt -Czb
```

## Benchmark

```bash
# Optmized model benchmark in sync mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'sync' -d CPU -m mobilenet_v1.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4
# Optmized model 20 parallel (-nireq 20) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m mobilenet_v1.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4 -nireq 20
# Int8 Quantizd model 56 parallel (-nireq 56) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m mobilenet_v1_i8.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt
/nvm/val_resized/ -b 4 -nireq 56
```
