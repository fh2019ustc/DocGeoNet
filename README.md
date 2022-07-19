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
The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html), [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet), and [DocProj](https://github.com/xiaoyu258/DocProj). Thanks for their wonderful works.
