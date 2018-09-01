#### Tag estimation 

##### Base Line

- Naive approach with VGG feature extractor + embedding matrix (동렬) (ongoing)
- Dual path network for TOP-100 tags (동렬) (ongoing)



#### Face2anime 

##### Data processing

- Facial landmarks data processing for pix2pix & CycleGAN (민기) (completed)

##### Pix2Pix

- Facial landmarks-to-anime image translation (민기) (ongoing)
- Conditional facial landmarks-to-anime image translation (동렬) (ongoing)
  - Landmarks + hair color + eye color + skin color in RGB format

##### CycleGAN

- Face-to-anime image translation (민기) (ongoing)
  - Face image + facial landmarks 



요약 

오동렬: Top 100 태그 분류기 학습중, 이미지 + 머리, 눈, 피부색 정보를 이용한 pix2pix 구현

조민기: 얼굴 키포인트 정보를 이용한 pix2pix 데이터 처리 및 pix2pix, cycleGAN 구현