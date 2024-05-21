<!--
 * @Description: This rtdetr_pytroch has be modified to detection spine xray.
 * @version: 
 * @Author: ThreeStones1029 2320218115@qq.com
 * @Date: 2024-04-16 02:22:36
 * @LastEditors: ShuaiLei
 * @LastEditTime: 2024-04-16 15:01:56
-->
## TODO
<details>
<summary> see details </summary>

- [x] Training
- [x] Evaluation
- [x] Predict
- [x] Export onnx
- [x] Upload source code
- [x] Upload weight convert from paddle, see [*links*](https://github.com/lyuwenyu/RT-DETR/issues/42)
- [x] Align training details with the [*paddle version*](../rtdetr_paddle/)
- [x] Tuning rtdetr based on [*pretrained weights*](https://github.com/lyuwenyu/RT-DETR/issues/42)

</details>



Notes
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.
- `url`<sup>`*`</sup> is the url of pretrained weights convert from paddle model for save energy. *It may have slight differences between this table and paper*
<!-- - `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$ and $tensorrt\\_fp16$ mode -->

## Quick start

<details>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>


<details>
<summary>Data</summary>

- Download and extract COCO 2017 train and val images.
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
- Modify config [`img_folder`, `ann_file`](configs/dataset/coco_detection.yml)
</details>



<details>
<summary>Training</summary>

- Training on a Single GPU:

```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Training on Multiple GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```
</details>

<details>
<summary>Evaluation</summary>

- Evaluation on Single GPUs:
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint
```

- Evaluation on Multiple GPUs:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

</details>

<details>
<summary>Predict</summary>

```shell
python tools/infer.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint 
```
</details>

<details>
<summary>Export</summary>

```shell
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```
</details>




<details open>
<summary>Train custom data</summary>

1. set `remap_mscoco_category: False`. This variable only works for ms-coco dataset. If you want to use `remap_mscoco_category` logic on your dataset, please modify variable [`mscoco_category2name`](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/data/coco/coco_dataset.py#L154) based on your dataset.

2. add `-t path/to/checkpoint` (optinal) to tuning rtdetr based on pretrained checkpoint. see [training script details](./tools/README.md).
</details>
