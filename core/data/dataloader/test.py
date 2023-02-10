import os
def generator_list_of_imagepath(path):
    image_list = []
    for image in os.listdir(path):
        # print(path)
        # print(image)
        if not image == '.DS_Store' and 'tif' == image.split('.')[-1]:
            image_list.append(image)
    return image_list


if __name__ == '__main__':
    #path = r'C:\Users\51440\Desktop\WLKdata\googleEarth\train\images'
    path = r'C:\Users\51440\Desktop\WLKdata\WLKdata-1111\train\images-LS'
    list=generator_list_of_imagepath(path)
    print(list)
    #dataset = VOCJibutiSegmentation()