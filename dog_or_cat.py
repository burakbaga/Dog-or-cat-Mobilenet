import numpy as np
from tensorflow.python.keras.applications import mobilenet
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image

mobile = mobilenet.MobileNet()

train_path = 'C:\\Users\\bagat\\OneDrive\\Masaüstü\\DCimages\\Train'
valid_path = 'C:\\Users\\bagat\\OneDrive\\Masaüstü\\DCimages\\Valid'
test_path = 'C:\\Users\\bagat\\OneDrive\\Masaüstü\\DCimages\\Test'

# verdiğimiz yoldan batch halinde resimleri alır
# aldığımız resimleri 0 ile 1 arasına sıkıştırmamız gerekiyor
train_batches = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input).flow_from_directory(train_path,
                                                                                                          target_size=(
                                                                                                          224, 224),
                                                                                                          batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input).flow_from_directory(valid_path,
                                                                                                          target_size=(
                                                                                                          224, 224),
                                                                                                          batch_size=10,
                                                                                                          )
test_batches = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input).flow_from_directory(test_path,
                                                                                                         target_size=(
                                                                                                         224, 224)
                                                                                                         ,batch_size=10,
                                                                                                         shuffle=False)

imgs, labels = next(train_batches)
print(train_batches.class_indices)

x=mobile.layers[-6].output #son 5 katmanı almadık
predictions=Dense(2,activation='softmax')(x)#output katmanı ekledik

model=Model(inputs=mobile.input,outputs=predictions)

for layer in model.layers[:-5]:# son 5 katmana kadar olanlar eğitime girmez
    layer.trainable=False

model.summary()

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# buraya batch halinde alınan resimleri veriyoruz
# steps per epoch her epochta kaç adımda verilerin tümünü alıcak toplam veri sayısı bölü batch boyutu
model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=10)

test_images, test_labels = next(test_batches)
print(test_batches.class_indices)  # cat 0. indis dog 1. indis

# test_labels = test_labels[:, 0]
# print(test_labels)
Categories = ['Cat', 'Dog']

predictions = model.predict_generator(test_batches, steps=1)

for i in predictions:
    print(i)


#saving model
model.save('dog_or_cat_mobilnet.h5')

def prepare_image(file):
    img=image.load_img(file,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array_expanded_dims=np.expand_dims(img_array,axis=0)
    return mobilenet.preprocess_input(img_array_expanded_dims)#imagei 0 ile 1 arasında normalize eder



predictions2 = model.predict(prepare_image('robin.jpg'))
print(predictions2)
index = int(np.argmax(predictions2))
print(Categories[index])
