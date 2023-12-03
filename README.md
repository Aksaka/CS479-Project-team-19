# CS479-Project-team-19
Diffusion-Based Video Synthesis from 3D Sparse Frames

![TMDM](https://github.com/Aksaka/CS479-Project-team-19/assets/56462998/d83d4d82-06cc-4999-90a0-57005f75d1bb)

## Time Morph Diffusion Model

  The exponential growth in video data prompts the need for efficient 3D video synthesis. In pursuit of this, we introduced the Time Morph Diffusion Model, envisioning a seamless transition between sparse frames for enhanced storage efficiency. Our project addresses the challenge of 3D video synthesis, aiming to reduce storage requirements. However, faced with challenges in the model's training, we were unable to achieve the desired results. We attribute these problems to limitations in GPU capacity for learning, lack of understanding of diffusion models, and problems with approaches. Moreover, due to challenges in model training, meaningful results with the Time Morph Diffusion Model could not be attained. Implementation of the Generalizable NeRF Transformer(GNT), remains unfulfilled at this stage leaving as a future work.

## How to start training TMDM

Install the all the requirements

```
pip install -r requirements.txt
```

Before you start, move to the project directory.
```
cd TMDM
```

Start the training TMDM by 

```
python main.py
```

(Before you start training, make sure that your dataset for training exists in the project directory)

## Pretrained model and datasets

You can download the pre-trained model from [here](https://drive.google.com/file/d/1_oYcH9FWaSJOpoqGVR80jYVW9EeYg9Ue/view?usp=drive_link).

(Link: https://drive.google.com/file/d/1_oYcH9FWaSJOpoqGVR80jYVW9EeYg9Ue/view?usp=drive_link)


And also you can download the dataset that we used in the project from [here](https://drive.google.com/file/d/1_rAVE77hT_bklQBG2haxttSz8gbLP4J-/view?usp=drive_link) 


(Link: https://drive.google.com/file/d/1_rAVE77hT_bklQBG2haxttSz8gbLP4J-/view?usp=drive_link)

## Result example

![Result](https://github.com/Aksaka/CS479-Project-team-19/assets/56462998/9c19f5f2-cfcd-4a4e-8631-8e69d3b9c2c4)
