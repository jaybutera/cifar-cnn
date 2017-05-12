from keras.datasets import cifar10
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

model = Sequential()
model.add( Conv2D(32, (5,5), input_shape=(32,32,3), activation='relu') )
model.add( MaxPooling2D(pool_size=(2,2)) )
model.add( Dropout(.1) )
model.add( Conv2D(16, (5,5), activation='relu') )
model.add( MaxPooling2D(pool_size=(2,2)) )
model.add( Dropout(.1) )
model.add( Flatten() )
model.add( Dense(128, activation='relu') )
model.add( Dense(10, activation='relu') )

model.compile(loss='categorical_crossentropy', optimizer='Adam')
model.summary()

model.fit(X_train, Y_train, batch_size=32, epochs=5)

print(model.evaluate(X_test, Y_test))
