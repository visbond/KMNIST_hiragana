This program does Japanese character recognition (Hiragana) from the KMNIST dataset, using deep learning. The PyTorch library is utilised. KMNIST is mainly for the handwritten (and no longer in current use) Kuzushiji script, but it also has 10 Hiragana characters with 7000 samples per class. This is the dataset we will be using.

Progressively more complex (and accurate) models are made.

1.  The first model "NetLin" simply computes a linear function of the pixels in the image, followed by log softmax. 

   It can be run by typing (if running for the first time, it will download the dataset first):
   
   ```
python3 hira_main.py --net lin
   ```

   It displays the accuracy and confusion matrix at the end. The best accuracy obtained with this simple net is around 70%. 
   
   Note that the rows of the confusion matrix indicate the target character, while the columns  indicate the one chosen by the network. (0="o", 1="ki", 2="su", 3="tsu", 4="na", 5="ha", 6="ma", 7="ya", 8="re", 9="wo"). 

   For examples of each character, please refer to the Kuzushiji paper, uploaded in this repository. This gives an insight into why some characters are often mistaken for another due to visual similarity. A basic listing of the characters is also given in hiragana.png in this repo.

   

2. The second net "NetFull" is a fully connected 2-layer network, using tanh at the hidden nodes and log softmax at the output node. 

   It can be run by:

   ```
   python3 hira_main.py --net full
   ```

   It achieves ~85% accuracy, a great improvement on the former net. The confusion matrix is also displayed at the end. We see that some characters are still consistently mistaken for other characters.

   

3. The final model is a convolutional network, "NetConv", with two convolutional layers plus one fully connected layer, all using relu as their activation function, followed by the output layer. 

   It can be run by typing:

   ```
   python3 hira_main.py --net conv
   ```

   The network consistently achieves 93% accuracy on the test set after 10 training epochs. 

   

4. Adding one more layer to the CNN net of the previous model gets us up to 94% accuracy, but increases training time. This model is called "NetConv_Custom".