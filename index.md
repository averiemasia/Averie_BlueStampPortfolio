# Project Name Here
Replace this text with a brief description (2-3 sentences) of your project. This description should draw the reader in and make them interested in what you've built. You can include what the biggest challenges, takeaways, and triumphs from completing the project were. As you complete your portfolio, remember your audience is less familiar than you are with all that your project entails!

You should comment out all portions of your portfolio that you have not completed yet, as well as any instructions:
```HTML 
<!--- This is an HTML comment in Markdown -->
<!--- Anything between these symbols will not render on the published site -->
```

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Averie M | Schreiber High School | Computer Science | Incoming Senior

**Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.**

![Headstone Image](logo.svg)
  
# Final Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/F7M7imOVGug" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE



# Second Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/Saz_tLSkUOA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Continuing working in colab, for my second milestone I advanced into making image detection a working game.  Trying to address some issues I saw in my first milestone with correctly identifying the images, I looked back through the original data set (that was used for training the model) and figured out how to most efficiently crop the image that my camera was taking so that my hand would be similar to the hands that trained the machine learning model.  In the future I want to work on making the code flow more smoothly and also giving the computer the option to cheat to win.  The original project I selected had used the pi camera to track the movements of the player's hand and accurately guess what move they are going to make, so I want to include a flare of that in my final project.

# First Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/iWbenZ6Ey8c" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

The original project I selected would've included a raspberry pi and a pi camera; however, technical difficulties led me to switch gears and move to working solely with google colab.  My original objective when choosing this project was to learn about machine learning and be able to apply training a model to a data set other than the Rock, Paper, Scissors images.  Although these issues are not ideal, I know that they are a huge part of the engineering process, especially if I decide to continue into Computer Science.  For this first milestone I was able to connect my camera to google colab and correctly identify whether the picture I took was Rock, Paper, or Scissors and though when I selected the project I thought I'd be using the command line to code on the raspberry pi, coding with python on google colab has still been enriching.  

# Schematics 
Here's where you'll put images of your schematics. [Tinkercad](https://www.tinkercad.com/blog/official-guide-to-tinkercad-circuits) and [Fritzing](https://fritzing.org/learning/) are both great resoruces to create professional schematic diagrams, though BSE recommends Tinkercad becuase it can be done easily and for free in the browser. 

# Code
Here's where you'll put your code. The syntax below places it into a block of code. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize it to your project needs. 

```c++
void setup() {
  // put your setup code here, to run once:
  ## set up code
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

## taking a picture
def takePic():
  from IPython.display import Image
  try:
    filename = take_photo()
    
  except Exception as err:
    # Errors will be thrown if the user does not have a webcam or if they do not
    # grant the page permission to access it.
    print(str(err))

## cropping the image
def cropImg():
  global im
  import cv2
  from matplotlib import pyplot as plt
  import matplotlib.pyplot as plt
  from scipy import ndimage
  import numpy as np

  def crop(img, center, width, height):
      return cv2.getRectSubPix(img, (width, height), center)

  im = cv2.imread("photo.jpg")
  im = crop(im, (320, 200), 330,330)

  plt.imshow(im)

## identifying and labeling the image
def identify():
  global im
  from numpy.ma.extras import row_stack
  import numpy as np
  import tensorflow as tf
  import cv2
  from matplotlib import pyplot as plt

  def crop(img, center, width, height):
      return cv2.getRectSubPix(img, (width, height), center)

  im = cv2.resize(im, (150, 150))
  im = im[...,::-1]

  plt.imshow(im)

# Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="/content/drive/MyDrive/Colab Notebooks/model.tflite")
  interpreter.allocate_tensors()

# Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

# Test model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array([im], dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])
  # print(output_data)
  global move
  plt.imshow(im)
  choice = max(output_data[0])
  if choice==output_data[0][0]:
    move = 'rock'
  elif choice==output_data[0][1]:
    move = 'paper'
  else:
    move = 'scissors'
  print ('YOUR MOVE:')
  print (move)

### logistical functions
import random
uwin == 0
## random selection of move
def playFair():
  #randomly chooses cmove
  global uwin
  global move
  comp = random.randint(0,2)
  if comp == 0:
    cmove = 'rock'
  elif comp == 1:
    cmove = 'paper'
  else:
    cmove = 'scissors'
  print ('MY MOVE:')
  print (cmove)
  #win statement
  if move == 'rock':
    if cmove == 'rock':
      print ('draw!')
    elif cmove == 'paper':
      print ('you lose! sorry!')
    else:
      print ('you win!')
      uwin += 1
  if move == 'paper':
    if cmove == 'paper':
      print ('draw!')
    elif cmove == 'scissors':
      print ('you lose! sorry!')
    else:
      print ('you win!')
      uwin += 1
  if move == 'scissors':
    if cmove == 'scissors':
      print ('draw!')
    elif cmove == 'rock':
      print ('you lose! sorry!')
    else:
      print ('you win!')
      uwin += 1
  return uwin
## cheating computer
def beatPlayer():
  global move
  # cheat to choose cmove
  if move == 'rock':
    cmove = 'paper'
  elif move == 'paper':
    cmove = 'scissors'
  else:
    cmove = 'rock'
  print('MY MOVE:')
  print(cmove)
  # win statements
  print('you lose! sorry!')
}

void loop() {
  // put your main code here, to run repeatedly:
while True:
# call on functions
  takePic()
  cropImg()
  identify()

# honest or cheating machine?
  if uwin < 5:
    playFair()
  else:
    beatPlayer()

  print('SCORE:')
  print(uwin)
}
```

# Bill of Materials
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
|  |  |  |  |
|  |  |  |  |


# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.
