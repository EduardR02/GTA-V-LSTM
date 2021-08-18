# GTA V LSTM
# Model
This project combines a CNN, which is used as a feature extractor  
with an LSTM. The CNN is an on image net pretrained imported network, like  
in my case InceptionV3, but could be any network, like Efficientnet (Transfer learning).  
I used Inception because Effiecientnet was about 6 times slower to train,  
and Inception + LSTM already took ~5 hours freezing the CNN + ~10 hours after unfreezing.

# Try it out
The data preprocessing, data collection and Neural Net can be used for  
anything where the input are images or video and the output is clearly defined,  
like in my case the steering inputs for the car.  
You can experiment with resolution, sequence length, learning rate, dropout etc. and find out  
what works best. In my case input resolution was 160x120.  

I have included a model trained by me if you want to try it out yourself quickly.  
The training data consisted of me driving fast and overtaking, so don't expect  
it to drive perfectly between the lines and slow like the other NPC cars.  
The link to download the pretrained model (too big for github):  
https://drive.google.com/drive/folders/1zQaXs__m-XiPH_OiMsFZ3zTV_7VscQM8?usp=sharing

# How to use
To collect training data or make predictions with the model you need to  
put the game in windowed mode with a resolution of 800x600  
(you could modify it but aspect ratio should stay the same),  
and put it at the top left corner of your monitor (better double check alignment).  
You can of course change this, or even should if you want to use the code for a different purpose.

Some parts of this project (like the screen capture, key input and similar) are from  
this sentdex tutorial:  
https://github.com/Sentdex/pygta5

# Demo
Here i disabled traffic and let the model drive, keep in mind that it has been trained to drive fast and overtake (4x sped up):  
![alt text][gif1]  
  
This and the above gif cover a large part of the highway around the whole map.  
In this particular drive the model only crashed once at the very end.  
![alt text][gif2]  
  
This is with traffic enabled, the model tries to overtake and not crash. I inluded some crashes to show how it reacts (2x sped up):  
![alt text][gif3]  

# Conlusions
Before this i had tried to train my own model based on sentdex' tutorial, where the model was only a CNN.  
I found the LSTM approach to work a lot better, especially when crashing or overtaking,
the downside being that it took a lot longer to train.  
One problem that I found was that i did not include a "do nothing" output,
making it very difficult for the model to regulate speed, as braking is very rare.  

[gif1]: https://github.com/EduardR02/GTA-V-LSTM/blob/main/gifs/first_drive_no_cars_4x.gif
[gif2]: https://github.com/EduardR02/GTA-V-LSTM/blob/main/gifs/second_drive_no_cars_4x.gif
[gif3]: https://github.com/EduardR02/GTA-V-LSTM/blob/main/gifs/with_cars_30fps.gif
