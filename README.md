# Truth-in-Noise: Spiral Weld Defect Detection under X-ray Imaging  Degradation via Diffusion Models

# DIWDD

## Train

```bash
cd DIWDD

accelerate launch train.py \
  --config config/diwdd_512x512.yaml \
  --results_folder output/trian_result \
  --result_name bs=16 \
  --num_epoch=150 \
  --batch_size 8 \
  --gradient_accumulate_every=1 \
  --val_every_epochs 10
```

## Sample

```bash
accelerate launch sample.py \
  --config config/diwdd_512x512.yaml \
  --results_folder output/test_result \
  --result_name xxx \
  --checkpoint output/trian_result/best_result/model-best.pt \
  --num_sample_steps 5 \
  --target_dataset NEUSW1700 \
  --batch_size 1
```

### Partial segmentation visualization of the NEU-SW-1700 dataset：
[[Uploading visualize.pdf…]()](https://github.com/ykzZldX7/DiWDD/blob/main/visualize.pdf)

We will release the NEU-SW-1700 dataset after the paper is accepted.




