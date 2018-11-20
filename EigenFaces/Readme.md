# Face Recognition: Let's go back to the basics
Face Recognition has been an active area of research for many decades now. One of the earliest documented attempts at using a computer program to recognize a human face dates all the way back to 1964-1966 by [Bledsoe and colleagues](https://archive.org/details/firstfacialrecognitionresearch) working for an unnamed intelligence agency in the US. Today, we have  [methods that are better than humans at recognizing faces!](https://arxiv.org/abs/1404.3840)

Most of the success in face recognition seen over the recent years can be directly attributed to the progress made in neural networks and deep learning. But, to understand the problem and the approaches taken today, it helps to start with the basics. In this post, we look closely at one of the early ideas to find success with face recognition - eigenfaces. It's a simple yet extremely elegant technique proposed by [Sirovich and Kirby](https://www.researchgate.net/publication/19588504_Low-Dimensional_Procedure_for_the_Characterization_of_Human_Faces) in 1987 and first applied to face recognition by [Turk and Pentland](https://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf) in 1991.

### Eigen Faces
To uniquely distinguish one face from another, you need a way to describe each face. This description is commonly referred to as a *feature(s)*. One of the popular approaches back then was to detect individual features corresponding to different regions in the face - eyes, mouth, nose, etc., and use the relationship between them to model the differences among faces. More often than not these features were handcrafted by someone looking at a large database of faces. Turk and Pentland approached the problem differently; they argued that it may not be necessary for features to have any direct relationship with our intuitive notion of the important regions in faces (eyes, ears, nose, etc.). Instead, they wanted to find out the features by reformulating the problem from an information theory perspective.

### Problem Formulation
*I know this is a distraction but bear with me.* Before solving any problem, one of the most important steps is to formulate the exact question that we want to answer. Big breakthroughs often appear simple in hindsight because of this... they often change the question on its head making the problem significantly easier.

Coming back to FR, how did the popular approaches at the time model their question? They often started with the assumption that they knew what features were useful for face-recognition (eyes, nose, mouth, etc) and they wanted to find the **relationship between them** such that they uniquely described a face, i.e,

> *How can we model the relationship between facial features to learn a model that can uniquely identify a person?*

Turk and Pentland, however, questioned the assumption made on the features above and instead asked - **What makes a good feature?** They took inspiration from information theory, where the goal is that of representation -

> *How can we represent (encode) a message (information) with as little data (bytes,space,storage) as possible in order to be able to successfully reconstruct (decode) the message minimal loss (of information)?*

While it might seem like a completely unrelated problem, there is a connection if you were looking to find the **most important features** in a face. One could argue that if we learned a compact representation of faces, condensed to the bare essentials that could then be used to sufficiently reconstruct them, then that condensed representation must contain the most important features in the face (after all, losing all other information didn't matter in the reconstruction process!). *Perspective is everything.* So, Turk and Pentland reformulated the problem -

> *How can we extract information contained in faces, such that, they can be encoded as efficiently as possible and compared against one another?*

### Approach
Reformulating the problem from the information theory perspective meant that there were now a lot of well-developed tools at their disposal to solve the problem. They used the most commonly used tool for dimensionality reduction - Principal Component Analysis(PCA).

Okay great! But, what does this mean and how do we use it? Let's start with an example: given bunch images of a set of people, how do we find a unique way to describe each of them?

![](barack.jpg)

Ever wondered how images are displayed on your screen? Sure, the computer looks at a bunch of 0s and 1s. But, each pixel in fact is represented as a number which indicates the level of intensity between 0-255.

![](grayscale.png)

If it's a black and white image, what you get then is a matrix `I` of size `w x h`, where `w` and `h` are the dimensions of the image in pixels. And each pixel's value represents the corresponding shade of gray. If it's a color image, then you'll have 3 such values for each pixel representing the particular red, blue and green components.





Here are the key ideas proposed by Turk and Pentland:

* Consider an image **I** (N x M pixels) , to be a point in a high-dimensional space (R<sup>N x M</sup>). Images of faces, being similar overall, can be expected to not be randomly distributed, but by a relatively small subspace. Let's call this Face space.

* To effectively model/estimate the face-subspace, we could use Principal Component Analysis to find the set of vectors that best represent the variation in a given distribution of faces (training samples/faces). The eigenvectors so obtained, will thus be called EigenFaces.

* Any face image can then be reconstructed as a linear weighted combination of the EigenFaces. We can compare faces by projecting them onto the face subspace and measuring the Euclidean distance between them.


### Revisiting Principal Component Analysis
<div class="alert alert-success">
**Note:** If you remember how to do PCA from your linear algebra class, you can skip this section. <br>

On the other hand, if you have forgotten what eigenvectors and eigenvalues are, here's a nice [video](https://www.youtube.com/watch?v=PFDu9oVAE-g) explaining it.
Also, here's a nice [blog](http://setosa.io/ev/principal-component-analysis/) on visualizing PCA.
</div>

Principal Component Analysis, also known as Discrete Karhunen Leove transformation is a technique often used to perform dimensionality reduction. Say what? It basically helps transform a bunch of data points existing in some high-dimensional (R<sup>d</sup>) space to a bunch of data points in fewer dimensions (R<sup>k</sup>,  k << d) while still maintaining the inherent relationship between them.

More precisely, it is a statistical tool that allows one to identify the principal directions in which the data varies from it’s covariance matrix. Wait... what is a covariance matrix? Let's recall -

**Variance:** is a measure of how far a set of data points are spread out from their mean.

**Co-variance:** is a measure of how two or more variables vary together.

The **covariance matrix** of a d-dimensional dataset, represents the variance and covariance between the dimensions. It is a symmetric matrix of size d x d such that, element at position (i,j)  represents the covariance between the i<sup>th</sup> and the j<sup>th</sup> dimensions and the diagonal represents the variance along each dimension.

Using the SVD (Singular value decomposition), compute it's eigenvectors and eigenvalues. *<-- more on SVD here.*


### Method

**Training** *<-- Need to expand this section with explanation, code or images/ step-wise results*
* Preprocessing: Detect faces & align them using any algorithm
* Centered Data Matrix: Compute the mean face and subtract it from all the faces in the dataset.
* Perform PCA to get the eigenvalues (aka eigenfaces)
* Select the top k eigen faces to be used for reconstruction (One natural way to do this is to use as many eigenvectors it takes to explain	p% of	the	data variability).
* Transform all faces in the dataset to the subspace based on the selected eigenfaces (one linear weight vector per identity) => W (kxn) weight Matrix

And we are done.

**Testing** *<-- Need to expand this section with explanation, code or images/ step-wise results*
* Given an input image I, subtract the mean obtained in step X from the Image
* Use the eigenvectors to transform this image to obtain a new weight vector.
* Iterate through all the stored weight vectors from step Y to identify the closest neighbor (by Euclidean distance). The face belonging to the identity X of the closest neighbor is our predicted label.

### Advantages
* Efficient data representation
* Easy to compute
* Compact (less storage thanks to dimensionality reduction)
* Globally optimal solution
* Robust to image distortions (e.g. rotation, provided this is represented in your original data)

### Disadvantages
* PCA is optimal for reconstruction, but may not be for discrimination. That is, the features learned may not be optimal to differentiate one person from another.
* Selection of the number of eigen faces is important. Fewer eigenfaces mean more information loss (as can be seen in the notebook).

Some of the disadvantages are addressed with LDA based facial recognition also known as Fisher Face. But more on that later. I hope this was a practical introduction to Face Recognition using eigen faces! If you run the notebook on LFW sample dataset, then you can expect ~75% accuracy in predicting the face presented in an image. Pretty neat for such a simple approach eh? Well simple if you remember linear algebra from your sophomore year, but it's never too late!

If you have any questions/issues regarding the notebook/the article feel free to shoot me an email.
