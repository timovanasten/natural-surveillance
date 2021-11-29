# Measuring Natural Surveillance at Scale: An Automated Methodology for Investigating the Relation Between the "Eyes on the Street" and Urban Safety

This repository contains the code belonging to my master thesis: [_Measuring Natural Surveillance at Scale: An Automated Methodology for Investigating the Relation Between the "Eyes on the Street" and Urban Safety_](https://github.com/timovanasten/natural-surveillance/blob/doc/doc/Measuring%20Natural%20Surveillance%20at%20Scale.pdf).
See the document for a detailed descripton of the project and the obained results.

![front-cover](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/front-cover.png?raw=true)

## Abstract
To create safe urban areas, it is important to gain insight into what influences the (perceived) safety of our cities and human settlements. One of the factors that can contribute to safety is the way urban spaces are designed. Previous work has highlighted the importance of _natural surveillance_: a type of surveillance that is a byproduct of how citizens normally and routinely use the environment. However, studying this concept is not a trivial task. Manual approaches such as observation studies are costly and time-consuming and have therefore often limited themselves to smaller geographical areas. 

In this work, we present a methodology that can automatically provide an estimate of natural surveillance by detecting building openings (i.e. windows and doors) in street level imagery and localizing them in 3 dimensions. The proposed method is able to estimate natural surveillance at the street segment level, while simultaneously being able to gather data on a whole city in a matter of hours. We then apply our method to the city of Amsterdam to analyze the relationship between natural surveillance and urban safety using the Amsterdam Safety Index. 

We conclude that our chosen operationalization of natural surveillance (road surveillability and occupant surveillability) is correlated with decreases in high impact crime and nuisance as well as increases in perceived safety. Furthermore we provide evidence for the existence of a threshold after which extra natural surveillance is no longer associated with higher degrees of safety.

## How to Use
This project is written and tested in Python 3.9.
### 1. Clone the repo and install dependencies
```sh
git clone https://github.com/timovanasten/natural-surveillance.git
cd natural-surveillance
pip install -r requirements.txt
```

### 2. Download the facade labeling model
The models used for detecting building openings within the street view imagery can be downloaded [here]( https://drive.google.com/drive/folders/1TfeIcQ8KlEvP1-ewGcTaj3SqU_IpoLUv). Then run
```sh
mkdir ./labeling/heatmap_fusion/model
```
and place ```resnet18_model_latest.pth.tar``` into the ```natural-surveillance/labeling/heatmap_fusion/model``` directory.

### 3. Google Cloud API key
Obtain a API key for the Google Street View Static API and add it to ```.secrets.yaml```. 
To not overshoot available credit, enter your credit in dollar in ```data/gsv_budget.txt```

## Dataset
![dataset](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/neighborhood-level.png?raw=true)
A dataset in CSV format with the estimated natural surveillance scores for over 6500 street segments spread over 43 neighborhoods in Amsterdam can be found in the ```data/results``` directory.


## Contact
For any questions, feel free to contact me on timovanasten@gmail.com.


## Visualizations

![data-collection](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/data-collection.png?raw=true)
![opening-localization](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/opening-localization-overview.png?raw=true)
![opening-detection](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/opening-detection.png?raw=true)
![localization-output](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/localization-output.png?raw=true)
