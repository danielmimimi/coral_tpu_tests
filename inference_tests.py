import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import time

# WHEN USED CUSTOM OP

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

model_nr = 1

script_dir = pathlib.Path(__file__).parent.absolute()
if model_nr == 0 :
    model_file = os.path.join(script_dir, 'models/mobilenet_v2_1.0_224_quant_edgetpu.tflite')
elif model_nr == 1:
    model_file = os.path.join(script_dir, 'models/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite')
    edgetpu_delegate = load_delegate('libedgetpu.so.1')
    posenet_decoder_delegate = load_delegate(os.path.join(script_dir,'posenet_lib/posenet_decoder.so'))
    interpreter = Interpreter(model_file, experimental_delegates=[edgetpu_delegate, posenet_decoder_delegate])
    interpreter.allocate_tensors()
elif model_nr == 2:
    model_file = os.path.join(script_dir, 'models/openvino_model_full_integer_quant_edgetpu.tflite')
else :
    model_file = os.path.join(script_dir, 'models/mobilenet_v2_1.0_224_quant_edgetpu.tflite')

if model_nr != 1 :
    # Initialize the TF interpreter
    interpreter = edgetpu.make_interpreter(model_file)
    interpreter.allocate_tensors()  

label_file = os.path.join(script_dir, 'imagenet_labels.txt')
image_file = os.path.join(script_dir, 'images/persons.bmp')




# Resize the image
size = common.input_size(interpreter)
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
amount_of_inferences = 100
start_time = time.time()
for i in range(amount_of_inferences) :
    common.set_input(interpreter, image)
    interpreter.invoke()
print("--- %s seconds ---" % ((time.time() - start_time)/amount_of_inferences))
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(label_file)
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))