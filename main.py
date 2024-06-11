from models.uNet import *
from data import *

'''
Refer to https://github.com/zhixuhao/unet/blob/master/main.py
'''

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,    #旋转角度
                        width_shift_range=0.05, #宽度偏移
                        height_shift_range=0.05,#高度偏移
                        shear_range=0.05,       #剪切强度
                        zoom_range=0.05,        #缩放强度
                        horizontal_flip=True,   #水平翻转
                        fill_mode='nearest')    #填充模式,‘nearest'表示用最近的像素填充
    myGene = trainGenerator1(2, './Crag/train','image','label',data_gen_args,save_to_dir = None, image_color_mode = "rgb",mask_color_mode = "grayscale")

    model = unet()

    model_checkpoint = ModelCheckpoint('unet_membrane2.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])

    testGene = testGenerator("./Crag/test/image")
    results = model.predict_generator(testGene,40,verbose=1)

    saveResult("./Crag/test/predict",results)