---
layout: post
comments: false
#title:  "Grad-CAM: Making Off-the-shelf Deep Models Transparent through Visual Explanations"
title: "Yes, Deep Networks are great, but are they Trust-worthy?"
#title: "Deep Networks look right, but do they look at the right regions?"
date:   2017-01-21 22:00:00
mathjax: true
---

* TOC
{:toc}

## Introduction

Convolutional Neural Networks (CNNs) and other deep networks have enabled unprecedented breakthroughs in a variety of Computer Vision tasks, ranging from Image Classification (classify the image into a category from a given set of categories), to semantic segmentation (segment the detected category), image captioning (describe the image in natural language), and more recently, visual question answering (answer a natural language question about an image). Despite their success, when these systems fail, they fail spectacularly disgracefully, without any warning or explanation, leaving us staring at an incoherent output, wondering why it said what it said. Their lack of decomposability into intuitive and understandable components makes them extremelly hard to interpret.

## Need for Transparent Interpretable Systems

A bit of history. Sometime in the 1980s, the [Pentagon wanted to harness computer technology to identify military tanks](https://neil.fraser.name/writing/tank/). They had a team which went and collected about 100 photographs of tanks hiding behind trees, and 100 photographs of trees with no tanks. A group of researchers trained a Neural Network to distinguish between scenes with and without tanks. Their Neural Net achieved 100% accuracy on their held out test set. When these spectacular results were presented at a conference, a person from the audience raised a concern about the training data they collected. After further investigation it turned out that all the images with tanks were taken on a cloudy day, and all images without tanks were taken on a sunny day. So, at that time the US Government was a proud owner of a multi-billion dollar computer that could tell you whether it was cloudy or not.

Similarly, with the [first fatality happening with a Tesla autonomous car](www.pbs.org/newshour/rundown/driver-killed-in-self-driving-car-accident-for-first-time/) a few months back, and with [Uber's self driving cars seen running red light](www.wired.com/2016/12/ubers-self-driving-car-ran-red-light-san-francisco/), everyone is looking for clear-cut answers/explanations as to why such mistakes happened and what steps are taken in-order to prevent those from happening again.

These examples remind us that what works on current benchmarks (or test beds) might not work well on real life systems. The day when machines take life decisions for us is not very far ahead.

> If we humans cannot look at predictions made by AI systems, and tell if they are catastrophic failures or brilliant decisions, we will not have any control when these get shipped out into products that impact human lives.

According to a [recent paper presented at ICML'16 workshop](https://arxiv.org/abs/1606.08813), a new regulation is coming out in Europe that states that “*whenever human subjects have their lives significantly impacted by an automatic decision making machine, the human subject has the right to know why the decision is made- i.e. right to explanation*”. This is supposed to turn into a law as early as 2018. So this isn't just an academic exercise anymore!

Let's now look at some scenarios where this sort of transparency and interpretability can help. Take the case where an AI system is significantly weaker than humans (eg. visual question answering), the goal of transparency and explanations would be to identify failure modes, which can help researchers to focus their efforts on the most fruitful research directions. Going one level up, when the AI system is on par with humans and is reliably deployable, (e.g., image classification and self-driving cars to some extent), the goal is to establish trust with end users using this application. Going to the highest level where the AI does significantly better than well trained humans (e.g. chess or Go), the goal of transparency and explanations can be machine teaching – i.e., a machine teaching a human on how to make accurate decisions. 

In order to build trust in intellegent systems and move towards their meaningful integration into our everyday lives, it is clear that we must build ‘transparent’ models that explain themselves.

---------

## Conventional approaches to Interpretability

For simplicity let's call the deep neural network function $$f$$.


<center><img src="http://i.giphy.com/yZxqmcIBSyCrK.gif" style="width: 800px;" ></center>

Passing an input $$ x $$  (say an image) to this function would output a probability distribution over a set of labels/categories, $$ y $$. Typically, this function is highly non-linear, due to the existence of non-linear activations (for eg. ReLU) interspersed in between compounded linear functions (convolutions or fully-connected layers).

### Backpropagation

Using the First-order Taylor-series approximation, we can approximate this non-linear function $$ f $$ as a linear function,

$$ f(x) = f(x_0) + (x-x_0) f'(x - x_0)$$

This now permits us to use gradients. Gradients by their definition indicate the rate of change of a function ($$ f $$ here), with respect to the variable ($$ x $$ here) surrounding an infinitesimally small region near that particular point. In our case, gradients can tell us how changes in $$ x $$ affect $$ y $$. In the interpretability line you can think of it like, how does changing a pixel in the input image change the network's behaviour for that input.

For example let us take an image of a cat, as shown below. Visualizing the gradient of the loss (for category cat) wrt the input pixels gives,
http://i.imgur.com/xs2sCC5.png
<center><img src="http://i.imgur.com/xs2sCC5.png" style="width: 400px;" ></center>

As we can see, this is pretty noisy.

### Deconv and Guided Backprop

[Deconvolution](https://arxiv.org/abs/1311.2901), and [Guided-backpropagation](https://arxiv.org/abs/1412.6806) modify the backward pass of ReLU which is well explained in this figure below:

<center><img src="http://i.imgur.com/bHLF8it.jpg" style="width: 600px;" ></center>

This results in much cleaner results,

<center><img src="http://i.imgur.com/FXxnSt4.png" style="width: 400px;" ></center>

**Why are simple gradients noisy, and gradients with Guided backprop aren't?**

In Guided-Backprop, during ReLU backward in addition to what is suppressed in simple backprop, we additionally suppress the negative gradients. Negative gradients at a particular ReLU neuron, state that this neuron has a negative influence on the class that we are trying to visualize.

During backpropagation there are paths that have positive influence and some that have negative influence, and these end up cancelling out in a weird interference pattern, causing gradients to seem noisy. Whereas in Guided Backpropagation, we only keep paths that lead to positive influence on the class score, and supress the ones that have negative influence, leading to much cleaner looking images.

Let's now take a different image like one below,

<center><img src="http://i.imgur.com/LUDL1V5.jpg" style="width: 200px;" ></center>

As we can see, there are 2 categories here - dog and cat. Lets visualize regions important for each of these 2 categories using Guided-Backpropagation (GB).

<center><img src="http://i.imgur.com/LWlaTLS.png" style="width: 600px;" ></center>

This is bad.  The visualization is unable to distinguish between pixels of cat and dog. In other words, the **visualization is not class-discriminative.** 

So, approximating the whole network as a linear function didn't give us anything great. What if we visualize the final fully-connected layer (i.e. visualize the gradient of the loss wrt to the penultimate fully-connected layer activations)? Problem: It is not possible to visualize a scalar. We need a tensor to visualize. We will look at how Bolei Zhou's Class Activation Mapping uses this tensor product to interpret Image Classification CNNs.

### Class Activation Mapping

It is known that as the depth of a CNN increases, higher-level visual constructs are captured. Furthermore, convolutional layers naturally retain spatial information which is lost in fully-connected layers, so we can expect the last convolutional layers to have the best compromise between high-level semantics and detailed spatial information. The neurons in these layers look for semantic class-specific information in the image (say object parts). Knowing the importance of each neuron activation (feature maps) for a particular class of interest can help us better understand where the whole deep model is looking at. For example, to understand why a neural network would predict "person" for given image, we would ideally expect the neural network to recognize that the feature maps that looks for the hands, faces, legs, etc. are more important than other feature maps. 

We know that the activations till the last convolutional layers (feature maps) are tensors. If we had a network in which the convolutional layers were followed directly by a prediction layer without any fully connected layers, we would have exactly what we need - a tensor product which can easily be visualized. 

[CAM](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) does exactly this - modify the base network to remove all fully-connected layers at the end, and include a tensor product (followed by softmax), which takes as input the Global-Average-Pooled convolutional feature maps, and outputs the probability for each class. Note that this modification of architecture forces us to retrain the network.

<center><img src="http://i.giphy.com/26xBL7wCaRe6SqFDq.gif" style="width: 600px;" ></center>

The weights learned in the last tensor product layer correspond to the neuron importance weights, i.e. - importance of the feature maps for each class of interest.

<center><img src="http://i.imgur.com/ObMP5b3.jpg" style="width: 600px;" ></center>


>We have seen that CAM needs a simplified architecture, and hence has to be trained again. This can sometime lead to a decrease in accuracy.

Now I am sure you are thinking,
**Can we get these visualizations without changing the base model, and without *any* re-training?**

The underlying un-modified network has definitely learned to distinguish between different classes, implying that it recognizes which activation at which layer to give importance in-order to make the particular decision.

Let us see how Grad-CAM uncovers these importance weights without any training.

## **Grad-CAM**- Gradient-weighted Class Activation Mapping

<center><img src="http://i.imgur.com/JaGbdZ5.png" style="width: 800px;" ></center>

To obtain the class-discriminative localization map, Grad-CAM computes the gradient of $$ y^c $$ (score for class c) with respect to feature maps $$ A $$ of a convolutional layer, i.e. $$ \frac{\partial y^c}{\partial A^k_{ij}}$$. These gradients flowing back are global-average-pooled to obtain the importance weights $$ \alpha{}_{k}^c $$:

$$ \alpha{}_{k}^c = \overbrace{\frac{1}{Z}\sum_{i}\sum_{j}}^{\text{global average pooling}}\underbrace{\vphantom{\sum_{i}\sum_{j}} \frac{\partial y^c}{\partial A_{ij}^{k}}}_{\text{gradients via backprop}} $$

>In most deep learning frameworks, this can be computed using just a single backward call till the convolutional layer.

This weight $$ \alpha{}_{k}^c $$ represents a partial linearization of the deep network downstream from $$ A $$, and captures the '*importance*' of feature map $$ k $$ for a target class $$ c $$. In general, $$ y^c $$ need not be the class score produced by an image classification CNN, and could be any differentiable activation. 
> To be concrete even though we introduced Grad-CAM using the notion ‘class’ from image classification (e.g., cat or dog), **visual explanations can be considered for any differentiable node in a computational graph, including words from a caption or the answer to a question, etc.**.

Similar to CAM, Grad-CAM heat-map is a weighted combination of feature maps, but followed by a ReLU:

$$ L_{\text{Grad-CAM}}^{c} = ReLU \underbrace{\left(\sum_k \alpha{}_{k}^{c} A^{k}\right)}_{\text{linear combination}} $$

> Notice that this results in a coarse heat-map of the same size as the convolutional feature maps ($$ 14 \times 14 $$ in the case of last convolutional layers of VGG and AlexNet networks). 

If the architecture is already CAM compatible – the weights learned in CAM are precisely the weights computed in Grad-CAM. Other than the ReLU, this makes **Grad-CAM a generalization of CAM**. This generalization is what allows Grad-CAM to be applicable to ***any CNN-based architecture***. 

<center><img src="http://i.imgur.com/4CKwYOR.jpg" style="width: 600px;" ></center>


### **Guided Grad-CAM**
While Grad-CAM visualizations are class-discriminative and localize relevant image regions well, they lack the ability to show fine-grained importance like pixel-space gradient visualization methods (Guided Backpropagation and Deconvolution). For example take the case of the left image in the above figure, Grad-CAM can easily localize the cat region; however, it is unclear from the low-resolutions of the heat-map why the network predicts this particular instance is ‘tiger cat’. In order to combine the best aspects of both, we can fuse Guided Backpropagation and the Grad-CAM visualizations via a pointwise multiplication. GradCAM overview figure above illustrates this fusion. 

This results in visualizations like below,

<center><img src="http://i.imgur.com/BbTL40i.jpg" style="width: 600px;" ></center>

This visualization is both high-resolution (when the class of interest is ‘tiger cat’, it identifies important ‘tiger cat’ features like stripes, pointy ears and eyes) and class-discriminative (it shows the ‘tiger cat’ but not the ‘boxer (dog)’).

## Demo
Time to test it out:

A live demo on Grad-CAM applied to image classification can be found at [gradcam.cloudcv.org/classification](gradcam.cloudcv.org/classification). 

Here is a quick video showing some of its functionalities.
<center>
<iframe width="756" height="455" src="https://www.youtube.com/embed/COjUB9Izk6E?start=160&end=198&" frameborder="0" allowfullscreen></iframe>
</center>
So, go ahead try it out with images of your interest, and let us know your comments/suggestions.

## Going beyond classification

Grad-CAM being a strict generalization to CAM lets us generate visual explanations from CNN-based models that cascade convolutional layers with more complex interactions. We will apply Grad-CAM to “beyond classification” tasks and models that utilize CNNs, like image captioning and Visual Question Answering (VQA).

## Image-Captioning

Lets try and visualize a simple Image captioning model (without attention) using Grad-CAM. We are going to build on top of the [publicly available 'neuraltalk2'](github.com/karpathy/neuraltalk2) implementation by Karpathy, that uses a finetuned VGG-16 CNN for images and an LSTM-based language model. Similar to the classification case, we can compute Grad-CAM for any user given caption. Given a caption, we compute the gradient of its log probability w.r.t. units in the last convolutional layer of the CNN (conv5_3 for VGG-16) and generate Grad-CAM visualizations. Lets look at some results below.

<center><img src="http://i.imgur.com/hGOJv2r.jpg" style="width: 800px;" ></center>

For first example, the Grad-CAM maps for the generated caption localizes every occurrence of both the kites and people inspite of their relatively small size. In the next example, see how Grad-CAM correctly highlights the pizza and the man, but ignores the woman nearby, since ‘woman’ is not mentioned in the caption.

Let's look at more examples with Guided Grad-CAM visualizations too,

<center><img src="http://i.imgur.com/nzsDorc.png" style="width: 800px;" ></center>


### Time to try our Grad-CAM Captioning Demo
<center>
<iframe width="756" height="455" src="https://www.youtube.com/embed/COjUB9Izk6E?start=90&end=153&" frameborder="0" allowfullscreen></iframe>
</center>
Check its limits at [gradcam.cloudcv.org/captioning](gradcam.cloudcv.org/captioning).

## Visual Question Answering

Lets visualize 2 VQA models - a simple baseline model, and a complicated Res-Net-based hierarchical co-attention model.

**Visualizing a simple VQA model without attention:**
We visualize a publicly available standard [VQA implementation by Jiasen Lu](github.com/VT-vision-lab/VQA_LSTM_CNN). It consists of a CNN to model images and a RNN (Recurrent Neural Network) language model for questions. The image and the question representations are fused to predict the answer with a 1000-way classification. Since this is a classification problem, lets pick an answer (the score $$ y_c $$) and use its score to compute Grad-CAM to show image evidence that supports the answer.

Below are some example visualizations for the VQA model trained with 3 different CNNs - AlexNet, VGG-16 and VGG-19. Even though the CNNs were not finetuned for the task of VQA, it is interesting to see how Grad-CAM helps understand these networks better by providing a localized high-resolution visualization of the regions the model is looking at. 

<center><img src="http://i.imgur.com/FczOO7b.jpg" style="width: 1000px;" ></center>

Notice in the first row of the above figure, for the question, “Is the person riding the waves?”, the VQA model with AlexNet and VGG-16 answered “No”, as they concentrated on the person mainly, and not the waves. On the other hand, VGG-19 correctly answered “Yes”, and it looked at the regions around the man in order to answer the question. In the second image, for the question, “What is the person hitting?”, the VQA model trained with AlexNet answered “Tennis ball” just based on context without looking at the ball. Such a model might be risky when employed in real-life scenarios. It is difficult to determine the trustworthiness of a model just based on the predicted answer. Grad-CAM visualizations provide an accurate way to explain the model’s predictions and help in determining which model to trust, without making any architectural changes or sacrificing accuracy. Notice in the last row of the above figure, for the question, “Is this a whole orange?”, the model looks for regions around the orange to answer “No”.

**Visualizing Res-Net-based VQA model with attention:**

[Lu et al. 2016](https://arxiv.org/abs/1606.00061) uses a 200 layer [ResNet](https://arxiv.org/abs/1512.03385) to encode the image, and jointly learn a hierarchical attention mechanism based on parses of question and image.

<center><img src="http://i.imgur.com/UNx369c.jpg" style="width: 800px;" ></center>

> Note that these networks were trained with no explicit attention mechanism enforced.

### Time to try our Grad-CAM VQA Demo

A live demo of Grad-CAM applied to a simple VQA model is hosted at  [gradcam.cloudcv.org/vqa](gradcam.cloudcv.org/vqa). Play with it and let us know what you think.

Here is an example:
<center>
<iframe width="756" height="455" src="https://www.youtube.com/embed/COjUB9Izk6E?start=17&end=86&" frameborder="0" allowfullscreen></iframe>
</center>

It is interesting to see that **common CNN + LSTM models are pretty good at localizing discriminative input regions despite not being trained on grounded image-text pairs**.

## Negative Explanations with Grad-CAM

Let's look at a new explanation modality - **negative explanations**. 

Using a slight modification to Grad-CAM we can obtain negative explanations, which highlight the support of the regions that would make the network predict a different class. These can be used to instill trust in an end user, in the sense that the underlying model does understand the class of interest, and it doesn't get confused because of other distracting classes.

This is done by negating the gradient of $$ y^c $$ (score for class $$ c $$) with respect to feature maps $$ A $$ of a convolutional layer. Thus the importance weights $$ \alpha{}_{k}^{c} $$ now become, 

$$ \alpha{}_{k}^c = \overbrace{\frac{1}{Z}\sum_{i}\sum_{j}}^{\text{global average pooling}}\underbrace{ \vphantom{\sum_{i}\sum_{j}} -\frac{\partial
y^c}{\partial A_{ij}^{k}} }_{\text{Negative gradients}} $$

<center><img src="http://i.imgur.com/2XgD1GM.png" style="width: 600px;" ></center>

----------

## Conclusions

In this blog post we have seen why it is important to have transparent machines which explain their decision making process. We looked at Gradient-based methods and CAM for interpreting these deep models. We saw that these methods either are not class-discriminative, or high-resolution, or require a modification to the model, which leads to a decrease in accuracy. We discussed a recently proposed alternative, Grad-CAM which solves the above mentioned problems. We saw its application to various Image Classification, Image Captioning and Visual Question Answering models. We believe that a true AI system should not only be intelligent, but also be able to reason about its beliefs and actions for humans to trust it.

This blog post is based on our paper, "**Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization**". It can be found in Arxiv via [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391). The paper has more experiments, including qualitative and quantitative comparisons to existing approaches. It also includes human-evaluations which show that Guided Grad-CAM explanations help users establish trust in the predictions made by deep networks. Another interesting aspect we find is that Guided Grad-CAM helps untrained users successfully discern a ‘stronger’ deep network from a ‘weaker’ one even when both networks make identical predictions, simply on the basis of their different explanations.

**Code:**
Grad-CAM only requires a couple of lines be added to your code. Our Torch implementation for Grad-CAM can be found at [github.com/ramprs/grad-cam/](github.com/ramprs/grad-cam/). 


<center><img src="http://i.imgur.com/W5qyE4A.png" style="width: 900px;" ></center>

Tensorflow implementation by Ankush, can be found at [github.com/Ankush96/grad-cam.tensorflow](https://github.com/Ankush96/grad-cam.tensorflow), and a Keras implementation by Jacobgil can be found at [github.com/jacobgil/keras-grad-cam](https://github.com/jacobgil/keras-grad-cam). Jacob also has a nice [blogpost on vehicle steering angle visualization using Grad-CAM](https://jacobgil.github.io/deeplearning/vehicle-steering-angle-visualizations).

**Webpage:** Here is a webpage where we will post grad-cam related updates.

----------

## Acknowledgements

I would like to thank my advisors Devi Parikh and Dhruv Batra without whom this work wouldn't have been possible. I would also like to thank my fellow teammates, Abhishek Das, Michael Cogswell, Ramakrishna Vedantam, Stefan Lee, and Harsh Agrawal for their insights and expertise that greatly assisted the research.

----------
