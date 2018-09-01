## Dependency 

Tensorflow-gpu 1.4 

NSML

Other package is listed in requirments.txt. run 

`pip install -r requirements.txt`

## Detail

You can find the detail explanation and exp result from the pdf file `1808nw_final.pdf` 

<img src=".\006_exp_result.JPG" width="400">


In addition,  you can check the all experiments that I went through from the excel file `1808nw_work.xlsx`

<img src=".\005_exp_log.JPG">



## Usage 

### Data Processing

1. **Face detector**  

   ```shell 
   python anime_face_detector.py \
     --data_path=/shared/data/nw_data/ \
     --save_path=/shared/data/meta/nw_data/detect/ \
     --draw_box True --write_list True
   ```

   `data_path`  has the input images in jpg or png format.

   You can activate two mode, `draw_box` and `write_list`. 

   `draw_box` detect the location of face, and draw the bounding box around the face and eyes. 

   `write_list` makes a numpy dictionary of the color of skin, hair and eye.

2. **Facial keypoints (PRNet)**

   In order to run following Image translation models, you need to produce 3 kinds of data. 1. list of key-points (.txt), 2. pix2pix data in RGB image (.jpg), 3. concatenated data in 4d array (.npy)

   1. **list of key-points** (jpg -> txt)

      ``` bash
      CUDA_VISIBLE_DEVICES=3 python -i demo.py \
      -i /shared/data/celeb_cartoon/celeb/ \
      -o /shared/data/meta/celeb  \
      --isKpt True --isShow True --isImage True 
      ```

   2. **pix2pix data in RGB image** (jpg -> txt)

      ``` bash
      python pix2pix_data_kpt.py \
      --data_path=/shared/data/meta/celeb/ \
      --save_path=/shared/data/meta/celeb/cycGAN_input/ \
      --meta_path=/shared/data/meta/celebmeta/
      ```

   3. **concatenated data in 4d array** 

      ```bash
      python cycGAN_data_kpt.py \
      --data_path=/shared/data/meta/celeb/ \
      --save_path=/shared/data/meta/celeb/cycGAN_input/ \
      --meta_path=/shared/data/meta/celebmeta/
      ```






### Image Translation 

1. **Pix2Pix (require the data 2-1, 2-2)**

    Trainning

    ```bash
    CUDA_VISIBLE_DEVICES=3 python pix2pix.py \
    --mode train \
    --output_dir anime_train \
    --max_epochs 200 \
    --input_dir anime/train \
    --which_direction BtoA \
    --checkpoint anime_train
    ```

    Testing 

    ```bash
    CUDA_VISIBLE_DEVICES=3 python pix2pix.py \
    --mode test \
    --output_dir anime_test \
    --input_dir anime/val \
    --checkpoint anime_train
    ```

   




2. **Pix2Pix-additional attribute** (require the data 2-1, 2-2, attribute list in 1)

   Run 

   ```bash
   python main.py --train True --dataDir=./data
   ```

   NSML version (Data is in anime_kpt)

   ```bash
   nsml run -e nsml_run.py -d anime_kpt
   ```

3. **Cycle GAN** (require the data 2-1, 2-3) 

    Training 

    ```bash
    CUDA_VISIBLE_DEVICES=3 python main.py \
    --dataset_dir=f2c_4dcyc --data_path=/shared/data/ \
    --input_nc=4 --output_nc=4 --continue_train=False \
    --checkpoint_dir=checkpoint
    ```

	Testing 
	
	```bash
	CUDA_VISIBLE_DEVICES=3 python main.py \
	--dataset_dir=f2c_4dcyc --data_path=/shared/data/ \
	--input_nc=4 --output_nc=4 --continue_train=False \
	--checkpoint_dir=checkpoint
	```


      The input data is in following format 

   ```
   /shared/data/face2cartoon/
   |-- testA
   |-- testB
   |-- trainA
   `-- trainB
   ```

   In my case, group A was celebA and groub B was anime character. 

    

   The `input_nc` and `output_nc` is the dimension of the input and output data. The default value is 3, so you only need this arguments when your data is not in RGB format. 

   

   Following is the NSML command
   `nsml run -e nsml_run.py -d face2anime`

   You should check the default argument setting before you launch this.

    
