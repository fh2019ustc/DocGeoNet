**Good news! Our new work exhibits state-of-the-art performances on the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset:
[DocScanner: Robust Document Image Rectification with Progressive Learning](https://arxiv.org/pdf/2110.14968v2.pdf)** 

**Good news! A comprehensive list of [Awesome Document Image Rectification](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification) methods is available.** 


# DocGeoNet


Any questions or discussions are welcomed!


## Inference 
1. Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1-OEvGQ36GEF9fI1BnAEHj_aByHorl7C7?usp=sharing), and put them to `$ROOT/model_pretrained/`.
2. Unwarp the distorted images in `$ROOT/distorted/`:
    ```
    python inference.py
    ```

## DIR300 Test Set
1. We release the [DIR300 test set](https://drive.google.com/drive/folders/1yySouQQ3BlH7OjnUhq4CLuvpX2KXtifX?usp=sharing) for evaluation.


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{feng2021doctr,
  title={Geometric Representation Learning for Document Image Rectification},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Wang, Yuechen and Li, Houqiang},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

## Acknowledgement
The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html), [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet), and [DocTr](https://github.com/fh2019ustc/DocTr). Thanks for their wonderful works.
