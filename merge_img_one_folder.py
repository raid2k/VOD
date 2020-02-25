import glob
import shutil
# 7,8,9차시 승차인원 데이터셋 중 jpg 확장자만 출력, resize 된 데이터는 사용하지 않음
# 하위 폴더에 나눠진 이미지를 한 폴더에 병합 
dir_list = [
            'C:/Users/gnt/Desktop/Codes/data/SOC_Center/09/original//**/*.jpg'
          
           ]

for pic_dir in dir_list:
    all_dir = glob.glob(pic_dir, recursive=True)
    for file in all_dir:
        print(file)
        shutil.copy(file,'C:/Users/gnt/Desktop/Codes/data/Img/')
