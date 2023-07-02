
# Neural Style Transfer

### A Brief overview 
Neural style transfer is a generative deep learning model that combines the contents of one image and the style of another.
Instead of traditional deep learning models that analyze a given image or set of input parameters to give an output, generative models
output new and original data that hasn't been fed to the model itself.

### Method
Instead of performing gradient descent to find an "accurate" solution,
generative models can be more challenging. How do we define loss if there's 
no "correct" answer?

#### Content loss
Since we want to preserve the contents of one image, we want to find a way to define the contents.
This can be done by using the L2 loss between an input image and a random noise. By iterating and minimizing 
the loss between the noise image and the input image, a copy of the key characteristics of the image is 
slowly developed.

#### Style Loss
Style loss can be found by taking a sample of the different filters activation values at different convolutional layers
inside of the model. Once those samples are taken, take the dot product between each filter. For a filter at index i,
and another at index j, the resultant value is stored in a matrix at i, j for a layer l. This new matrix
is called a gram matrix, and is meant to be similar to a covariance matrix of the different filters in different areas.
Similar to the content loss, we will take an input image for the style, and a noise image, and minimize the differences in the gram 
matrices. The generated image will take on the textures and some colour of the input image given enough time.

#### Total Loss
We just want to minimize our content and style differences from our two input images to generate a new output, therefore
we can write the total loss as a linear combination of the two different loss functions.

### Implementation
The project was then implemented on the convolutional layers of VGG19, and the layers were frozen in order to not alter 
the model's weights. The loss function was calculated during each epoch, and inference was ran on the generated image, style image,
and content images. The loss functions were then used to perform gradient descent on the generated image, using the Adam optimizer.

### Results
![alt text](https://github.com/torch360/neural_style_transfer/gen_images/kevin-card.png?raw=true)