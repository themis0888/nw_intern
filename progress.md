<<<<<<< HEAD
# Auto-tagging

## model
1. 일반적인 Image with attribute 모델에서는 [1] image feature extractor G(x) 뒤에 [2] image to tag mapping function F(x) 를 달아서 사용한다. [1]에서 높은 레벨의 semantic feature를 추출해내는 네트워크가 많아졌고, 이를 잘 embedding 해주는 F(x)를 찾는게 관건. 
2. 위 방법에 대한 차선책으로, G(x) 와 F(x) 를 합해서 F(G(x))를 학습시키는 방법도 있다. 다만 아래 2번 이슈로 인해 레이블의 질이 좋지 않기 때문에 학습에 방해요소가 될 수 있다. 
3. 보통의 문제라면 F(G(x))에 매핑된 포인트들이 잘 클러스터를 이루어 분포가 서로 분리되도록 하는 연구들이 많다. 가령 https://arxiv.org/pdf/1708.00938.pdf 이런게 그렇다.  그런데 우리의 문제는 레이블이 서로 체인을 갖는 경우가 많다. 
4. 

## Issue
1. Danbooru 데이터는 별다른 클래스 분류가 없기 때문에 [1] 부분을 일반적인 방법으로 학습할 수가 없다. 
2. Tag의 특징 관련해서 여러 이슈가 있다. 
	2.1 [Commentary_request], [Bad_id] 등의 전혀 관련 없는 태그들이 존재함 
	=> 수가 적다면 수작업으로 위와 같은 태그들을 걸러줄 수 있다. 
	2.2 [Hair], [Black hair] 처럼 Tag들이 서로 독립적이지 않다. 
	2.3 Tag가 성실하게 작성된 것이 아니다. 가령 [Balck hair]태그를 달고 있다면 높은 확률로 흑발이겠지만, [Black hair] 태그가 없다고 해서 흑발이 아닐지는 모른다. 
	=> Negative 에 대해서 label smoothing을 적용해주면 비교적 노이즈에 robust해진다. https://arxiv.org/pdf/1512.00567.pdf 7번항목 참고. 하지만 근본적인 해결책은 못된다. 
	2.4 
3. 

## Related works 
- About model #1 [1], [Classifying Disney Characters from Commercial Merchandise using  Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/265_Report.pdf)http://cs231n.stanford.edu/reports/2016/pdfs/265_Report.pdf



## Progress

7/16~7/22

- Data processing (Completed)
- Prototype 
	- Model (Dual path) (Completed)
	https://arxiv.org/pdf/1711.05535.pdf
	- Data feeding & Control (Completed)
	- 
=======
# Auto-tagging

## model
1. 일반적인 Image with attribute 모델에서는 [1] image feature extractor G(x) 뒤에 [2] image to tag mapping function F(x) 를 달아서 사용한다. [1]에서 높은 레벨의 semantic feature를 추출해내는 네트워크가 많아졌고, 이를 잘 embedding 해주는 F(x)를 찾는게 관건. 
2. 위 방법에 대한 차선책으로, G(x) 와 F(x) 를 합해서 F(G(x))를 학습시키는 방법도 있다. 다만 아래 2번 이슈로 인해 레이블의 질이 좋지 않기 때문에 학습에 방해요소가 될 수 있다. 
3. 보통의 문제라면 F(G(x))에 매핑된 포인트들이 잘 클러스터를 이루어 분포가 서로 분리되도록 하는 연구들이 많다. 가령 https://arxiv.org/pdf/1708.00938.pdf 이런게 그렇다.  그런데 우리의 문제는 레이블이 서로 체인을 갖는 경우가 많다. 
4. 

## Issue
1. Danbooru 데이터는 별다른 클래스 분류가 없기 때문에 [1] 부분을 일반적인 방법으로 학습할 수가 없다. 
2. Tag의 특징 관련해서 여러 이슈가 있다. 
	2.1 [Commentary_request], [Bad_id] 등의 전혀 관련 없는 태그들이 존재함 
	=> 수가 적다면 수작업으로 위와 같은 태그들을 걸러줄 수 있다. 
	2.2 [Hair], [Black hair] 처럼 Tag들이 서로 독립적이지 않다. 
	2.3 Tag가 성실하게 작성된 것이 아니다. 가령 [Balck hair]태그를 달고 있다면 높은 확률로 흑발이겠지만, [Black hair] 태그가 없다고 해서 흑발이 아닐지는 모른다. 
	=> Negative 에 대해서 label smoothing을 적용해주면 비교적 노이즈에 robust해진다. https://arxiv.org/pdf/1512.00567.pdf 7번항목 참고. 하지만 근본적인 해결책은 못된다. 
	2.4 
3. 

## Related works 
- About model #1 [1], [Classifying Disney Characters from Commercial Merchandise using  Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/265_Report.pdf)http://cs231n.stanford.edu/reports/2016/pdfs/265_Report.pdf



## Progress

7/16~7/22

- Data processing (Completed)
- Prototype 
	- Model (Dual path) (Completed)
	https://arxiv.org/pdf/1711.05535.pdf
	- Data feeding & Control (Completed)
	- 
>>>>>>> 25dde7d6bb745cc2942cf08c3a746dc7a1c7c407
- 