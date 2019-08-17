# Inception V3

2018_R5-2019_R1.1 OpenVINO Inception_v3

## Prerequests

### Env & datasets

1. OpenVINO 2018_R5 (currently installed in /home/ubuntu/intel/dldt)
2. Resize ImageNet val into val_bmp (299 * 299 * 3) (currently in /home/ubuntu/intel)
3. Export inception_v3.pb, and download ckpb
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

* `-nireq`: Async only. Number of inference requests (threads) in parallel. It should be no larger than core number.

* `-Czb`: required in `validation_app` and `calibration_tool`. "Zero is a background" flag. For models that use 0 as background, such as inception and mobilenet.

## Prepare OpenVINO model

### Pre-pare inception_v3.pb and inception_v3.ckpb (optional)

```bash
# inception_v3.ckpb
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

# inception_v3.pb
git clone https://github.com/tensorflow/models.git
python ~/models/research/slim/export_inference_graph.py --alsologtostderr --model_name=inception_v3 --output_file=./inception_v3.pb
```

### Model Optimization (Convert to OpenVINO IR)

```bash
# Convert from TensorFlow to OpenVINO model
/home/ubuntu/intel/dldt/model-optimizer/mo_tf.py --input_model inception_v3.pb --input_checkpoint inception_v3.ckpt -b 4 --reverse_input_channels --mean_values [127.5,127.5,127.5] --scale 127.5 --input=input --output=InceptionV3/Predictions/Reshape_1
# Validation acc with smalle dataset to ensure everything is correct
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/validation_app -m inception_v3.xml -i /home/ubuntu/intel/val_bmp_32/val.txt -Czb
```

### Calibration/Quantization (Convert IR to Int8 IR)

```bash
# Calibrate/quantize optmized model
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/calibration_tool -m inception_v3.xml -i /home/ubuntu/intel/val_bmp_32/val.txt -subset 32 -Czb
# prepare labels for quantizd model
cp inception_v3.labels inception_v3_i8.labels
# validate quantizd model
/home/ubuntu/intel/dldt/inference-engine/bin/intel64/Release/validation_app -m inception_v3_i8.xml -i /home/ubuntu/intel/val_bmp_32/val.txt -Czb
```

## Benchmark

```bash
# Optmized model benchmark in sync mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'sync' -d CPU -m inception_v3.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4
# Optmized model 20 parallel (-nireq 20) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m inception_v3.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt/nvm/val_resized/ -b 4 -nireq 20
# Int8 Quantizd model 56 parallel (-nireq 56) in async mode
/root/dldt/inference-engine/bin/intel64/Release/benchmark_app -api 'async' -d CPU -m inception_v3_i8.xml -l /root/dldt/inference-engine/bin/intel64/Release/lib/libcpu_extension.so -i /mnt
/nvm/val_resized/ -b 4 -nireq 56
```
