🚀 **Exciting update! We have created a demo for our paper on Hugging Face Spaces, showcasing the capabilities of our DocTr. [Check it out here!](https://huggingface.co/spaces/HaoFeng2019/DocGeoNet)**

🔥 **Good news! Our new work [DocTr++: Deep Unrestricted Document Image Rectification](https://github.com/fh2019ustc/DocTr-Plus) comes out, capable of rectifying various distorted document images in the wild.**

🔥 **Good news! Our new work exhibits state-of-the-art performances on the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset:
[DocScanner: Robust Document Image Rectification with Progressive Learning](https://drive.google.com/file/d/1mmCUj90rHyuO1SmpLt361youh-07Y0sD/view?usp=share_link)** with [Repo](https://github.com/fh2019ustc/DocScanner).

🔥 **Good news! A comprehensive list of [Awesome Document Image Rectification](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification) methods is available.** 


# DocGeoNet

<p>
    <a href='https://arxiv.org/pdf/2210.08161.pdf' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/spaces/HaoFeng2019/DocGeoNet' target="_blank"><img src='https://img.shields.io/badge/Online-Demo-green'></a>
</p>

> [**Geometric Representation Learning for Document Image Rectification**](https://arxiv.org/pdf/2210.08161.pdf)  
> ECCV 2022

Any questions or discussions are welcomed!


## 🚀 Demo [(Link)](https://huggingface.co/spaces/HaoFeng2019/DocGeoNet)
1. Upload the distorted document image to be rectified in the left box.
2. Click the "Submit" button.
3. The rectified image will be displayed in the right box.
4. Our demo environment is based on a CPU infrastructure, and due to image transmission over the network, some display latency may be experienced.

![image](https://github.com/fh2019ustc/DocGeoNet/assets/50725551/56e1742c-35c2-4dab-b965-5aa42c21c4c3)


## Training
- We train the network using the [Doc3D](https://github.com/fh2019ustc/doc3D-dataset) dataset.


## Inference 
1. Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1-OEvGQ36GEF9fI1BnAEHj_aByHorl7C7?usp=sharing), and put them to `$ROOT/model_pretrained/`.
2. Unwarp the distorted images in `$ROOT/distorted/` and output the rectified images in `$ROOT/rec/`:
    ```
    python inference.py
    ```

## DIR300 Test Set
1. We release the [DIR300 test set](https://drive.google.com/drive/folders/1yySouQQ3BlH7OjnUhq4CLuvpX2KXtifX?usp=sharing) for evaluation the rectification algorithms.


## Evaluation
- ***Important.*** In the [DocUNet Benchmark dataset](https://www3.cs.stonybrook.edu/~cvl/docunet.html), the '64_1.png' and '64_2.png' distorted images are rotated by 180 degrees, which do not match the GT documents. It is ingored by most of existing works. Before the evaluation, please make a check.
- Use the rectified images available from [Baidu Cloud](https://pan.baidu.com/s/1NsfdHmzFAf-_PBo2IPObaA?pwd=pd3d) for reproducing the quantitative performance on the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) reported in the paper and further comparison. We show the performance results of our method in the following table. For the performance of [other methods](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification), please refer to [DocScanner](https://github.com/fh2019ustc/DocScanner) and our paper.
- Use the rectified images available from [Google Drive](https://drive.google.com/drive/folders/1vQYGg-UvxZrvWYyqIEborFhokoUmg6iJ?usp=share_link) to reproduce the quantitative performance on the [DIR300 Test Set](https://drive.google.com/drive/folders/1yySouQQ3BlH7OjnUhq4CLuvpX2KXtifX?usp=sharing). For the performance of [other methods](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification), please refer to the paper.
- ***Image Metrics:*** We use the same evaluation code for MS-SSIM and LD as [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset based on Matlab 2019a. Please compare the scores according to your Matlab version. We provide our Matlab interface file at ```$ROOT/ssim_ld_eval_DocUNet.m``` and ```$ROOT/ssim_ld_eval_DIR300.m``` for the DocUNet and DIR300 Benchmark, respectively.
- ***OCR Metrics:*** The index of 30 document (60 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for our OCR evaluation is ```$ROOT/ocr_img_DocUNet.txt``` (*Setting 1*, following [DocTr](https://github.com/fh2019ustc/DocTr)). Please refer to [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) for the index of 25 document (50 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for their OCR evaluation (*Setting 2*). We provide the OCR evaluation code at ```$ROOT/OCR_eval_DocUNet.py``` and ```$ROOT/OCR_eval_DIR300.py``` for the DocUNet and DIR300 Benchmark, respectively. The version of pytesseract is 0.3.8, and the version of [Tesseract](https://digi.bib.uni-mannheim.de/tesseract/) in Windows is recent 5.0.1.20220118. 
Note that in different operating systems, the calculated performance has slight differences.

|      Benchmark Dataset     |      Method      |    MS-SSIM   |      LD     |     ED (*Setting 1*)    |       CER      |      ED (*Setting 2*)   |      CER     | 
|:----------------:|:----------------:|:------------:|:--------------:| :-------:|:--------------:|:-------:|:--------------:|
|      *DocUNet*   |      DocGeoNet   |     0.5040   |     7.71    |    379.00 |     0.1509     |    713.94 |     0.1821     | 

|      Benchmark Dataset     |      Method      |    MS-SSIM   |      LD     |     ED   |       CER      |
|:----------------:|:----------------:|:------------:|:--------------:| :-------:|:--------------:|
|      *DIR300*   |      DocGeoNet   |     0.6380   |     6.40    |    664.96 |     0.2189     |

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{feng2022docgeonet,
  title={Geometric Representation Learning for Document Image Rectification},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Wang, Yuechen and Li, Houqiang},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

```
@inproceedings{feng2021doctr,
  title={DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction},
  author={Feng, Hao and Wang, Yuechen and Zhou, Wengang and Deng, Jiajun and Li, Houqiang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={273--281},
  year={2021}
}
```

```
@article{feng2021docscanner,
  title={DocScanner: Robust Document Image Rectification with Progressive Learning},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Tian, Qi and Li, Houqiang},
  journal={International Journal of Computer Vision (IJCV)},
  year={2025}
}
```

## Acknowledgement
The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) and [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet). Thanks for their wonderful works.


## Contact
For commercial usage, please contact the email ([haof@mail.ustc.edu.cn](haof@mail.ustc.edu.cn)).
