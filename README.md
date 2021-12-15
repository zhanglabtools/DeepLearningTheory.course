# Deep Learning Theory

## About the Course
In recent years, deep learning has made great progress. It is widely used in many fields, such as computer vision, speech recognition, natural language processing, bioinformatics, etc., and also has a significant impact on basic science fields, such as applied and computational mathematics, statistical physics, computational chemistry and materials science, life science, etc. However, it should be noted that deep learning as a black-box model, its ability is explored through a large number of experiments. The theory of deep learning has gradually attracted the attention of many researchers, and has made progress in many aspects.

This course is closely around the latest development of deep learning theory. It intends to teach mathematical models, theories, algorithms and numerical experiments related to a series of basic problems from multiple perspectives of deep learning theory. This course is designed for doctoral, postgraduate and senior undergraduate students in all majors, who have basic knowledge of machine learning and deep learning.


The topics and the corresponding material are as follows:
  1. **Introduction to Deep Learning**  [material](#Introduction-to-deep-learning) [slides](./course_files/Lecture1.OverviewofDeepLearning.pdf)
  2. **Algorithmic Regularization** [material](#Algorithmic-Regularization) [slides](./course_files/Lecture2.AlgortihmicRegularization.pdf)
  3. **Inductive Biases due to Dropout** [material](#Inductive-Biases-due-to-Dropout) [slides](./course_files/Lecture3.InductiveBiasesduetoDropout.pdf)
  4. **Tractable Landscapes for Nonconvex Optimization** [material](#Tractable-Landscapes-for-Nonconvex-Optimization) [slides](./course_files/Lecture4.TractableLandscapes.pdf)
  5. **Multilayer Convolutional Sparse Coding** [material](#Multilayer-Convolutional-Sparse-Coding) [slides](./course_files/Lecture5.ML-CSC.pdf)
  6. **Vulnerability of Deep Neural Networks** [material](#Vulnerability-of-Deep-Neural-Networks) [slides](./course_files/Lecture6.VulnerabilityofDeepNeuralNetworks.pdf)
  7. **Information Bottleneck Theory** [material](#Information-Bottleneck-Theory) [slides](./course_files/Lecture7.InformationBottleneckTheoryofDNNs.pdf)
  8. **Neural Tangent Kernel** [material](#Neural-Tangent-Kernel) [slides](./course_files/Lecture8.NeuralTangentKernel.pdf)
  9. **Dynamic System and Deep Learning** [material](#Dynamic-System-and-Deep-Learning) [slides](./course_files/Lecture9.DynamicSystemandDeepLearning.pdf)
  10. **Dynamic View of Deep Learning** [material](#Dynamic-View-of-Deep-Learning) [slides](./course_files/Lecture10.DynamicViewofDeepLearning.pdf)
  11. **Approximation Theory of Neural Networks** [material](#Approximation-Theory-of-Neural-Networks)
  12. **Neural Networks Learn the Geodesic Curve in the Wasserstein Space** [material](#Neural-Networks-Learn-the-Geodesic-Curve-in-the-Wasserstein-Space)
  13. **Generative Model** [material](#Generative-Model)

##  Prerequisites

Mathematical Analysis, Linear Algebra, Mathematical Statistics, Numerical Optimization, Matrix Theory, Fundamentals of Machine Learning and Deep Learning

## Introduction to Deep Learning

### Key papers
+ Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
+ Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural networks, 2(5), 359-366.
+ Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.
+ Kramer, M. A. (1991). Nonlinear principal component analysis using autoassociative neural networks. AIChE journal, 37(2), 233-243.
+ Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4681-4690).
+ Lawrence, S., Giles, C. L., Tsoi, A. C., & Back, A. D. (1997). Face recognition: A convolutional neural-network approach. IEEE transactions on neural networks, 8(1), 98-113.
+ He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
+ Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).


## Algorithmic Regularization

### Key papers
+ Arora, S., Cohen, N., Hu, W., & Luo, Y. (2019). Implicit regularization in deep matrix factorization. Advances in Neural Information Processing Systems, 32, 7413-7424.
+ Byrd, J., & Lipton, Z. (2019, May). What is the effect of importance weighting in deep learning?. In International Conference on Machine Learning (pp. 872-881). PMLR.
+ Gunasekar, S., Lee, J., Soudry, D., & Srebro, N. (2018, July). Characterizing implicit bias in terms of optimization geometry. In International Conference on Machine Learning (pp. 1832-1841). PMLR.
+ Gunasekar, S., Woodworth, B., Bhojanapalli, S., Neyshabur, B., & Srebro, N. (2018, February). Implicit regularization in matrix factorization. In 2018 Information Theory and Applications Workshop (ITA) (pp. 1-10). IEEE.
+ Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., & Srebro, N. (2018). The implicit bias of gradient descent on separable data. The Journal of Machine Learning Research, 19(1), 2822-2878.
+ Xu, D., Ye, Y., & Ruan, C. (2021). Understanding the role of importance weighting for deep learning. arXiv preprint arXiv:2103.15209.


## Inductive Biases due to Dropout

### Key papers
+ Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1), 1929-1958.
+ Arora, R., Bartlett, P., Mianjy, P., & Srebro, N. (2021, July). Dropout: Explicit forms and capacity control. In International Conference on Machine Learning (pp. 351-361). PMLR.
+ Mianjy, P., Arora, R., & Vidal, R. (2018, July). On the implicit bias of dropout. In International Conference on Machine Learning (pp. 3540-3548). PMLR.
+ Baldi, P., & Sadowski, P. J. (2013). Understanding dropout. Advances in neural information processing systems, 26, 2814-2822.
+ Wager, S., Wang, S., & Liang, P. S. (2013). Dropout training as adaptive regularization. Advances in neural information processing systems, 26, 351-359.
+ Cavazza, J., Morerio, P., Haeffele, B., Lane, C., Murino, V., & Vidal, R. (2018, March). Dropout as a low-rank regularizer for matrix factorization. In International Conference on Artificial Intelligence and Statistics (pp. 435-444). PMLR.

## Tractable Landscapes for Nonconvex Optimization 

### Key papers
+ Du, S. S., Zhai, X., Poczos, B., & Singh, A. (2018, September). Gradient Descent Provably Optimizes Over-parameterized Neural Networks. In International Conference on Learning Representations.
+ Ge, R., Huang, F., Jin, C., & Yuan, Y. (2015, June). Escaping from saddle points—online stochastic gradient for tensor decomposition. In Conference on learning theory (pp. 797-842). PMLR.
+ Ge, R., Lee, J. D., & Ma, T. (2016, December). Matrix completion has no spurious local minimum. In Proceedings of the 30th International Conference on Neural Information Processing Systems (pp. 2981-2989).
+ Ge, R., Lee, J. D., & Ma, T. (2017). Learning one-hidden-layer neural networks with landscape design. arXiv preprint arXiv:1711.00501.
+ Hardt, M., Ma, T., & Recht, B. (2018). Gradient Descent Learns Linear Dynamical Systems. Journal of Machine Learning Research, 19, 1-44.
+ He, F., Wang, B., & Tao, D. (2019, September). Piecewise linear activations substantially shape the loss surfaces of neural networks. In International Conference on Learning Representations.

## Multilayer Convolutional Sparse Coding

### Key papers
+ Zhang, Z., & Zhang, S. (2021). Towards understanding residual and dilated dense neural networks via convolutional sparse coding. National Science Review, 8(3), nwaa159.
+ Papyan, V., Romano, Y., & Elad, M. (2017). Convolutional neural networks analyzed via convolutional sparse coding. The Journal of Machine Learning Research, 18(1), 2887-2938.
+ Papyan, V., Sulam, J., & Elad, M. (2017). Working locally thinking globally: Theoretical guarantees for convolutional sparse coding. IEEE Transactions on Signal Processing, 65(21), 5687-5701.
+ Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
+ Zeiler, M., Krishnan, D., Taylor, G., & Fergus, R. (2011). Deconvolutional Networks for Feature Learning. In Comput. Vis. Pattern Recognit.(CVPR), 2010 IEEE Conf (pp. 2528-2535).
+ He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
+ Pelt, D. M., & Sethian, J. A. (2018). A mixed-scale dense convolutional neural network for image analysis. Proceedings of the National Academy of Sciences, 115(2), 254-259.


## Vulnerability of Deep Neural Networks

### Key papers
+ Fawzi, A., Fawzi, H., & Fawzi, O. (2018). Adversarial vulnerability for any classifier. arXiv preprint arXiv:1802.08686.
+ Shafahi, A., Huang, W. R., Studer, C., Feizi, S., & Goldstein, T. (2018). Are adversarial examples inevitable?. arXiv preprint arXiv:1809.02104.
+ Li, J., Ji, R., Liu, H., Liu, J., Zhong, B., Deng, C., & Tian, Q. (2020). Projection & probability-driven black-box attack. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 362-371).
+ Li, Y., Li, L., Wang, L., Zhang, T., & Gong, B. (2019, May). Nattack: Learning the distributions of adversarial examples for an improved black-box attack on deep neural networks. In International Conference on Machine Learning (pp. 3866-3876). PMLR.
+ Wu, A., Han, Y., Zhang, Q., & Kuang, X. (2019, July). Untargeted adversarial attack via expanding the semantic gap. In 2019 IEEE International Conference on Multimedia and Expo (ICME) (pp. 514-519). IEEE.
+ Rathore, P., Basak, A., Nistala, S. H., & Runkana, V. (2020, July). Untargeted, Targeted and Universal Adversarial Attacks and Defenses on Time Series. In 2020 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.

## Information Bottleneck Theory

### Key papers


## Neural Tangent Kernel

### Key papers


## Dynamic System and Deep Learning

### Key papers


## Dynamic View of Deep Learning
### Key papers



## Approximation Theory of Neural Networks

## Neural Networks Learn the Geodesic Curve in the Wasserstein Space

## Generative Model




