# Cat vs Dog Identification
### A simple application of transfer learning using convolutinal neural nets

![Dog](readmeImage/monroePredict.png)

I used the prebuilt InceptionV3 network and added my own fully connected
layers. Then using the .ipynb, I built a few generators to shuffle images into
the network and train on the differences between cats and dogs. I froze all
layers save for the final connected layer. After training on a few thousand
images, I used a test data set to verify the model is working/generalizing. It
peforms at ~89% on out-of-training-sample images. Afterwards, I began building
out a very simple Flask app that allows the user to upload an image and then
have that image run through the model on the fly. It's still slow, as it has
to reload the network and resetablish the training weights each time. That
will need to be improved.

In the above image, I've shown the model's prediction for that picture of a
dog. The right output is the "dog-ness" and the left number is the "cat-ness."
It's very confident that that dog is a dog.

#### To Do
* Make it pretty
* Change how the network is loaded so it isn't so slow
* Show the compressed (changed to 200x200) image so the user can gauge if the
model sucks or if their image needs resizing

## Also, this happened:

![DuckHunt Dog](readmeImage/duckhuntDog.png)

* Class 0 (the left number) = Cat-ness
* Class 1 (the right number) = Dog-ness
* The model knows that's a dog. 

Original Dataset here: [https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)
