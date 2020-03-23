# TensorFlow Specializations

This repo contains two TensorFlow related specializations on Coursera: [TensorFlow in Practice](https://www.coursera.org/specializations/tensorflow-in-practice) and [TensorFlow: Data and Deployment](https://www.coursera.org/specializations/tensorflow-data-and-deployment).


## Courses

**TensorFlow in Practice**

- Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
  - A New Programming Paradigm
    > Welcome to this course on going from Basics to Mastery of TensorFlow. We're excited you're here! In week 1 you'll get a soft introduction to what Machine Learning and Deep Learning are, and how they offer you a new programming paradigm, giving you a new set of tools to open previously unexplored scenarios. All you need to know is some very basic programming skills, and you'll pick the rest up as you go along.
    You'll be working with code that works well across both TensorFlow 1.x and the TensorFlow 2.0 alpha. To get started, check out the first video, a conversation between Andrew and Laurence that sets the theme for what you'll study...
  - Introduction to Computer Vision
    > Welcome to week 2 of the course! In week 1 you learned all about how Machine Learning and Deep Learning is a new programming paradigm. This week you’re going to take that to the next level by beginning to solve problems of computer vision with just a few lines of code!
  - Enhancing Vision with Convolutional Neural Networks
    > Welcome to week 3! In week 2 you saw a basic Neural Network for Computer Vision. It did the job nicely, but it was a little naive in its approach. This week we’ll see how to make it better, as discussed by Laurence and Andrew here.
  - Using Real-world Images
    > Last week you saw how to improve the results from your deep neural network using convolutions. It was a good start, but the data you used was very basic. What happens when your images are larger, or if the features aren’t always in the same place? Andrew and Laurence discuss this to prepare you for what you’ll learn this week: handling complex images!

- Convolutional Neural Networks in TensorFlow
  - Exploring a Larger Dataset
    > In the first course in this specialization, you had an introduction to TensorFlow, and how, with its high level APIs you could do basic image classification, an you learned a little bit about Convolutional Neural Networks (ConvNets). In this course you'll go deeper into using ConvNets will real-world data, and learn about techniques that you can use to improve your ConvNet performance, particularly when doing image classification!
    In Week 1, this week, you'll get started by looking at a much larger dataset than you've been using thus far: The Cats and Dogs dataset which had been a Kaggle Challenge in image classification!
  - Augmentation: A technique to avoid overfitting
    > You've heard the term overfitting a number of times to this point. Overfitting is simply the concept of being over specialized in training -- namely that your model is very good at classifying what it is trained for, but not so good at classifying things that it hasn't seen. In order to generalize your model more effectively, you will of course need a greater breadth of samples to train it on. That's not always possible, but a nice potential shortcut to this is Image Augmentation, where you tweak the training set to potentially increase the diversity of subjects it covers. You'll learn all about that this week!
  - Transfer Learning
    > Building models for yourself is great, and can be very powerful. But, as you've seen, you can be limited by the data you have on hand. Not everybody has access to massive datasets or the compute power that's needed to train them effectively. Transfer learning can help solve this -- where people with models trained on large datasets train them, so that you can either use them directly, or, you can use the features that they have learned and apply them to your scenario. This is Transfer learning, and you'll look into that this week!
  - Multiclass Classifications
    > You've come a long way, Congratulations! One more thing to do before we move off of ConvNets to the next module, and that's to go beyond binary classification. Each of the examples you've done so far involved classifying one thing or another -- horse or human, cat or dog. When moving beyond binary into Categorical classification there are some coding considerations you need to take into account. You'll look at them this week!

- Natural Language Processing in TensorFlow
  - Sentiment in text
    > The first step in understanding sentiment in text, and in particular when training a neural network to do so is the tokenization of that text. This is the process of converting the text into numeric values, with a number representing a word or a character. This week you'll learn about the Tokenizer and pad_sequences APIs in TensorFlow and how they can be used to prepare and encode text and sentences to get them ready for training neural networks!
  - Word Embeddings
    > Last week you saw how to use the Tokenizer to prepare your text to be used by a neural network by converting words into numeric tokens, and sequencing sentences from these tokens. This week you'll learn about Embeddings, where these tokens are mapped as vectors in a high dimension space. With Embeddings and labelled examples, these vectors can then be tuned so that words with similar meaning will have a similar direction in the vector space. This will begin the process of training a neural network to udnerstand sentiment in text -- and you'll begin by looking at movie reviews, training a neural network on texts that are labelled 'positive' or 'negative' and determining which words in a sentence drive those meanings.
  - Sequence models
    > In the last couple of weeks you looked first at Tokenizing words to get numeric values from them, and then using Embeddings to group words of similar meaning depending on how they were labelled. This gave you a good, but rough, sentiment analysis -- words such as 'fun' and 'entertaining' might show up in a positive movie review, and 'boring' and 'dull' might show up in a negative one. But sentiment can also be determined by the sequence in which words appear. For example, you could have 'not fun', which of course is the opposite of 'fun'. This week you'll start digging into a variety of model formats that are used in training models to understand context in sequence!
  - Sequence models and literature
    > Taking everything that you've learned in training a neural network based on NLP, we thought it might be a bit of fun to turn the tables away from classification and use your knowledge for prediction. Given a body of words, you could conceivably predict the word most likely to follow a given word or phrase, and once you've done that, to do it again, and again. With that in mind, this week you'll build a poetry generator. It's trained with the lyrics from traditional Irish songs, and can be used to produce beautiful-sounding verse of it's own!

- Sequences, Time Series and Prediction
  - Sequences and Prediction
    > Hi Learners and welcome to this course on sequences and prediction! In this course we'll take a look at some of the unique considerations involved when handling sequential time series data -- where values change over time, like the temperature on a particular day, or the number of visitors to your web site. We'll discuss various methodologies for predicting future values in these time series, building on what you've learned in previous courses!
  - Deep Neural Networks for Time Series
    > Having explored time series and some of the common attributes of time series such as trend and seasonality, and then having used statistical methods for projection, let's now begin to teach neural networks to recognize and predict on time series!
  - Recurrent Neural Networks for Time Series
    > Recurrent Neural networks and Long Short Term Memory networks are really useful to classify and predict on sequential data. This week we'll explore using them with time series...
  - Real-world time series data
    > On top of DNNs and RNNs, let's also add convolutions, and then put it all together using a real-world data series -- one which measures sunspot activity over hundreds of years, and see if we can predict using it.

**TensorFlow: Data and Deployment**

- Browser-based Models with TensorFlow.js
  - Introduction to TensorFlow.js
    > Welcome to Browser-based Models with TensorFlow.js, the first course of the TensorFlow for Data and Deployment Specialization. In this first course, we’re going to look at how to train machine learning models in the browser and how to use them to perform inference using JavaScript. This will allow you to use machine learning directly in the browser as well as on backend servers like Node.js. In the first week of the course, we are going to build some basic models using JavaScript and we'll execute them in simple web pages.
  - Image Classification In the Browser
    > This week we'll look at Computer Vision problems, including some of the unique considerations when using JavaScript, such as handling thousands of images for training. By the end of this module you will know how to build a site that lets you draw in the browser and recognizes your handwritten digits!
  - Converting Models to JSON Format
    > This week we'll see how to take models that have been created with TensorFlow in Python and convert them to JSON format so that they can run in the browser using Javascript. We will start by looking at two models that have already been pre-converted. One of them is going to be a toxicity classifier, which uses NLP to determine if a phrase is toxic in a number of categories; the other one is Mobilenet which can be used to detect content in images. By the end of this module, you will train a model in Python yourself and convert it to JSON format using the tensorflow.js converter.
  - Transfer Learning with Pre-Trained Models
    > One final work type that you'll need when creating Machine Learned applications in the browser is to understand how transfer learning works. This week you'll build a complete web site that uses TensorFlow.js, capturing data from the web cam, and re-training mobilenet to recognize Rock, Paper and Scissors gestures.

- Device-based Models with TensorFlow Lite
  - Device-based models with TensorFlow Lite
    > Welcome to this course on TensorFlow Lite, an exciting technology that allows you to put your models directly and literally into people's hands. You'll start with a deep dive into the technology, and how it works, learning about how you can optimize your models for mobile use -- where battery power and processing power become an important factor. You'll then look at building applications on Android and iOS that use models, and you'll see how to use the TensorFlow Lite Interpreter in these environments. You'll wrap up the course with a look at embedded systems and microcontrollers, running your models on Raspberry Pi and SparkFun Edge boards.
  - Running a TF model in an Android App
    > Last week you learned about TensorFlow Lite and you saw how to convert your models from TensorFlow to TensorFlow Lite format. You also learned about the standalone TensorFlow Lite Interpreter which could be used to test these models. You wrapped with an exercise that converted a Fashion MNIST based model to TensorFlow Lite and then tested it with the interpreter.
    This week you'll look at the first of the deployment types for this course: Android. Android is a versatile operating system that is used in a number of different device type, but most commonly phones, tablets and TV systems. Using TensorFlow Lite you can run your models on Android, so you can bring ML to any of these device types. While it helps to understand some Android programming concepts, we hope that you'll be able to follow along even if you don't, and at the very least try out the full sample apps that we'll explore for Image Classification, Object Detection and more!
  - Building the TensorFLow model on iOS
    > The other popular mobile operating system is, of course, iOS. So this week you'll do very similar tasks to last week -- learning how to take models and run them on iOS. You'll need some programming background with Swift for iOS to fully understand everything we go through, but even if you don't have this expertise, I think this weeks content is something you'll find fun to explore -- and you'll learn how to build a variety of ML applications that run on this important operating system!
  - TensorFlow Lite on devices
    > Now that you've looked at TensorFlow Lite and explored building apps on Android and iOS that use it, the next and final step is to explore embedded systems like Raspberry Pi, and learn how to get your models running on that. The nice thing is that the Pi is a full Linux system, so it can run Python, allowing you to either use the full TensorFlow for Training and Inference, or just the Interpreter for Inference. I'd recommend the latter, as training on a Pi can be slow!

- Data Pipelines with TensorFlow Data Services
  - Data Pipelines with TensorFlow Data Services
    > You'll learn about the types of data that you would normally come across when doing machine learning.
  - Exporting your data into the training pipeline
    > This week you’re going to start looking at the code for using the data with input pipelines!
  - Performance
    > How you load your data into your model can have a huge impact on how efficiently the model trains. You'll learn how to handle your data input to avoid bottlenecks, race conditions and more!
  - Publishing your datasets
    > This week let’s learn about how you can share your data with the world in a way that’s easy for others to consume!

- Advanced Deployment Scenarios with TensorFlow
  - TensorFlow Extended
  - Sharing pre-trained models with TensorFlow Hub
  - Tensorboard: tools for model training
  - Federated Learning