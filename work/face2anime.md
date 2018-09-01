## Face2anime

#### Data processing

- Anime data - every images has no background but only foreground. 
- Facial information extraction from NAVER webtoon data (3D model, facial key-points, 3D projection) 
  - path = /shared/data/nw_data
- Anime character + facial data for pix2pix 
  - Sex classifier would help a lot 
- Anime character + facial data (concatenated) 
- Hair, eye, skin color estimator + face detector 
  - Currently, it can detect only one face, but you could do it for multiple face with minor fix





#### Base Line 

- Face2anime translation with Cycle GAN, UNIT, UGATIT, Fusion image generation
  - Cycle GAN : hardly distorted
  - UNIT : prone to memorize specific data pattern 
  - UGATIT : still learning
  - Fusion GAN : Not implemented yet. 
  - NSML & environment issue 
- Studying UNIT, MUNIT, UGATIT, Generating a Fusion Image (ongoing)



### Approach

------------------------------------------

#### Model_v1 

- Separate the distributions from two set on the latent space 
- Applying GAN model to replace the skip connection of the U-Net 
- Figure out that the size of latent variable Z is important s
- Shape learning 



#### Pix2Pix

- Key-point to image translation 





# Contribution 

## Data Processing  

#### Face Detector for Cartoon  

- This model detect the **face and attributes** from the animation data. The information it provide is the location of **face, eye, nose, mouth and the color of hair, eye and skin**.

- This model provide the **likelihood of the proposed location**, so user can manipulate the threshold.

- The downside of this model is it propose **only one position** for the location of the face. Therefore, the model detect only one face when it took a scene which has multiple character. 

- **For the NAVER webtoon data **

  - The detection rate is around **15%** over the entire webtoon scene where 30~40% of scenes does not have a face. 

  - It does work well for some webtoons in Japanese anime style. However, for the webtoon which is **far from the anime style**, it does not. 

  - 웹툰 그림체에 따라 성능이 매우 다르다. 
1. 얼굴조차 제대로 못 찾는 웹툰(참새~~) 
  ![014_08_not_a_face](https://media.oss.navercorp.com/user/10262/files/25f117c2-a7a0-11e8-92ba-321ff209d65b)

2. 얼굴은 찾는데 머리 눈 색이 정확하지 않은 웹툰 (신의 탑) 
  ![013_07_wrong_color](https://media.oss.navercorp.com/user/10262/files/2a67146e-a7a0-11e8-87ef-32be33f2cbb9)

3. 대체로 잘 찾는 웹툰. (신석기녀)
  ![013_03_color](https://media.oss.navercorp.com/user/10262/files/dbe663fa-a79d-11e8-901e-14f82cf1da8e)

   

- 의미? 

  - Available to obtain the massive amount of **face data from Danbooru**, and **Naver webtoon**.
  - In case of the Danbooru, the model shows the **solid performance** for the detecting eye, nose, mouth and face. It could be useful to apply this model on the other project that require the detail facial information, such like **sentimental analysis**, **facial expression estimation**.. 





### Facial Landmark 
![597447_13_cut_007](https://media.oss.navercorp.com/user/10262/files/c8e375fc-a7a4-11e8-8af3-3ad749aff4c9)
![597447_13_cut_007_skpt](https://media.oss.navercorp.com/user/10262/files/cdf9fd5e-a7a4-11e8-9e70-fbd3c2eea11b)
- PRNet can estimate the 68 facail landmarks even the part of face is blocked. However, you can apply this model only on the human-like cartoons because this model is trained with the human face data. 

  - In addition, 눈이 큰 캐릭터를 입력할 경우 얼굴 크기를 지나치게 크게 예측하는 경향이 강하다. 애니메이션 특성상 여성 캐릭터에서 이런 현상이 매우 잦게 발생하는데, 캐릭터의 남/여 구별이 가능해진다면 데이터의 질을 더 높일 수 있을 것이다. 

- There is two dataset with this tool 

  1. Facial key-points & animation image pair for Pix2Pix 
  2. 4D image data 

  

  

  

  1. 
로스 너무 크니까 디버깅 ㄱㄱ LS 간 켜보고 

