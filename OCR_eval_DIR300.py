def Levenshtein_Distance(str1, str2):
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1 
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    return matrix[len(str1)][len(str2)]

def cal_cer_ed(path_ours, tail='_rec'):
    path_gt='C:/Users/fengh/Desktop/DIR300/gt/'
    cer1=[]
    ed1=[]
    lis=[5,7,8,10,12,27,28,29,31,36,53,55,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,85,94,96]+\
         [103,107,108,111,115,126,128,129,130,133,135,139,140,148,149,151,159,160,161,162,163,164,165,166,167,169,170,173,174,177]+\
         [201,202,203,205,217,218,222,223,225,227,228,237,238,239,264,265,266,271,273,277,278,285,286,288,291,294,295,296,298,300]  # 90 images in DIR300
    print(len(lis))
    for i in range(1,301):
        if i not in lis:
            continue
        gt=Image.open(path_gt+str(i)+'.png')
        img1=Image.open(path_ours+str(i) + tail)
        content_gt=pytesseract.image_to_string(gt)
        content1=pytesseract.image_to_string(img1)
        l1=Levenshtein_Distance(content_gt,content1)
        ed1.append(l1)
        cer1.append(l1/len(content_gt))
    print('CER: ', np.mean(cer1))
    print('ED:  ', np.mean(ed1))

def evalu(path_ours, tail):
    cal_cer_ed(path_ours, tail)
