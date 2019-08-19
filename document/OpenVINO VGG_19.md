# OpenVINO vgg_19 Guide

## Prerequests

### Env & datasets

1. 2018_R5-2019_R1.1 (currently installed in /home/ubuntu/intel/dldt)
2. Resize ImageNet val into val_bmp (224 * 224 * 3) (currently in /home/ubuntu/intel)
3. Export vgg_19.pb, and download ckpb
4. val_bmp_32  and `val.txt`

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

### Pre-pare vgg_19.pb and vgg_19.ckpb (optional)

```bash
# vgg_19.ckpb
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz

# vgg_19.pb
# --labels_offset=1 is necessary
git clone https://github.com/tensorflow/models.git
python ~/models/research/slim/export_inference_graph.py --alsologtostderr --model_name=vgg_19 --labels_offset=1 --output_file=./vgg_19.pb
```

### Model Optimization (Convert to OpenVINO IR)

```bash
# Convert from TensorFlow to OpenVINO model
/home/ubuntu/intel/dldt/model-optimizer/mo_tf.py --input_model vgg_19.pb --input_checkpoint vgg_19.ckpt -b 4 --reverse_input_channels --mean_values [123.68,116.78,103.94]
# Validation acc with smalle dataset to ensure everything is correct
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/validation_app -m vgg_19.xml -i /home/ubuntu/intel/val_bmp_32/val.txt
```

**Note that Top 1 and Top 5 accuracy of vgg_19 should be close to 71.1% and 89.8%.**

### Calibration/Quantization (Convert IR to Int8 IR)

```bash
# Calibrate/quantize optmized model
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/calibration_tool -m vgg_19.xml -i /home/ubuntu/intel/val_bmp_32/val.txt -subset 32
# prepare labels for quantizd model
cp vgg_19.labels vgg_19_i8.labels
# validate quantizd model
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/validation_app -m vgg_19_i8.xml -i /home/ubuntu/intel/val_bmp_32/val.txt
```

## Benchmark

```bash
# Optmized model benchmark in sync mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'sync' -d CPU -m vgg_19.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4
# Optmized model 20 parallel (-nireq 20) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m vgg_19.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4 -nireq 20
# Int8 Quantizd model 56 parallel (-nireq 56) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m vgg_19_i8.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt
/nvm/val_resized/ -b 4 -nireq 56
```
