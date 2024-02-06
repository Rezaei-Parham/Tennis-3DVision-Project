import argparse
import Models , LoadBatches
from keras import optimizers
from keras.utils import plot_model

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--training_images_name", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 360  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--epochs", type = int, default = 1000 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "-1")
parser.add_argument("--step_per_epochs", type = int, default = 200 )

args = parser.parse_args()
training_images_name = args.training_images_name
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
step_per_epochs = args.step_per_epochs
optimizer_name = optimizers.Adadelta(lr=1.0)
modelTN = Models.TrackNet.TrackNet
m = modelTN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='categorical_crossentropy', optimizer= optimizer_name, metrics=['accuracy'])

if load_weights != "-1":
	m.load_weights("weights/model." + load_weights)
plot_model( m , show_shapes=True , to_file='TrackNet.png')
model_output_height = m.outputHeight
model_output_width = m.outputWidth
Generator  = LoadBatches.InputOutputGenerator( training_images_name,  train_batch_size,  n_classes , input_height , input_width , model_output_height , model_output_width)
for ep in range(1, epochs+1 ):
	print("Epoch :", str(ep) + "/" + str(epochs))
	m.fit_generator(Generator, step_per_epochs)
	if ep % 50 == 0:
		m.save_weights(save_weights_path + ".0")