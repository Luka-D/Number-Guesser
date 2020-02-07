from tkinter import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageGrab
import io
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Create Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#Train
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4)

#Verify
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)

#Save
model.save('num_reader.model')
model.save_weights('weights.h5')

#Recognize drawn number
new_model = tf.keras.models.load_model("num_reader.model")

predict = new_model.predict([x_test])

#
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
num_rows = 20
num_cols = 3
num_images = num_rows*num_cols
class_names = ["0","1","2","3","4","5","6","7","8","9"]
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predict[i], y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predict[i], y_test)
plt.tight_layout()
plt.show()

#Globals
lastx, lasty = 0, 0
width = 300
height = 300
line_width = 10
guess = ""

#Definitions
def xy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    global lastx, lasty
    for i in range(line_width):
        canvas.create_oval(lastx, lasty, event.x, event.y, fill='black', width=line_width)
    lastx, lasty = event.x, event.y

#Root Create + Setup
root = Tk()
f1 = Frame(root).grid(row=0,column=0)
f2 = Frame(root).grid(row=0,column=1)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

#Canvas Create + Setup
canvas = Canvas(f1,width=width,height=height)
canvas.grid(column=0, row=0, sticky=(N, W, E, S))
canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", addLine)

#Image Draw Copy
image1 = Image.new("RGB", (width, height), 0)
draw = ImageDraw.Draw(image1)

#Buttons
button1 = Button(f1,text="Guess", command=lambda : getter(canvas))
button2 = Button(f1, text='Clear', command=lambda : clear(canvas))
label = Label(f1,text='Guessed Number:')
anslabel = Label(f1,text=guess)
button1.grid(column=1, row=1)
button2.grid(column=2, row=1)
label.grid(column=3, row=1)
anslabel.grid(column=4,row=1)

#Getter
def getter(canvas):
    x=canvas.winfo_rootx()+canvas.winfo_x()
    y=canvas.winfo_rooty()+canvas.winfo_y()
    x1=x+canvas.winfo_width()
    y1=y+canvas.winfo_height()
    #ImageGrab.grab((x+80,y+80,x1+200,y1+200)).save(r"C:\\Users\\lukal\\Desktop\\cheese.jpg")
    img = ImageGrab.grab((x+80,y+80,x1+180,y1+180)).convert('L')
    
    #Attempt 2
    #t1 = img.convert("1")
    t2 = np.array(img).astype(np.uint8)
    t2 = np.invert(t2)
    t3 = cv2.resize(t2, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
    t4 = tf.keras.utils.normalize(t3, axis=1)
    t5 = np.expand_dims(t3, axis=0)

    plt.imshow(t2, cmap=plt.cm.binary)
    plt.show()
    plt.imshow(t3, cmap=plt.cm.binary)
    plt.show()
    plt.imshow(t4, cmap=plt.cm.binary)
    plt.show()
    
    #print("T1\n",t1)
    print("T2\n",t2)
    print("T3\n",t3)
    #print("T4\n",t4)
    new_model = tf.keras.models.load_model("num_reader.model")
    prediction = new_model.predict(t5, batch_size=None)
    pl = np.argmax(prediction)
    #guess = p1
    print(pl, prediction)
    
    #Second Frame
    fig = Figure(figsize=(6,6))
    pt = fig.add_subplot(111)
    pt.grid(False)
    pt.set_xticks(range(10),1.0)
    pt.set_yticks([])
    thisplot = plt.bar(range(10), prediction[0], color="#777777")
    predicted_label = np.argmax(prediction)
    thisplot[predicted_label].set_color('blue')
    canvas2 = FigureCanvasTkAgg(fig, master=f2)
    canvas2.draw()
    
#Clear
def clear(canvas):
    canvas.delete("all")
    
#Main Loop
root.mainloop()
