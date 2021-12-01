# Measuring Natural Surveillance at Scale: An Automated Methodology for Investigating the Relation Between the "Eyes on the Street" and Urban Safety

This repository contains the code belonging to my master thesis: [_Measuring Natural Surveillance at Scale: An Automated Methodology for Investigating the Relation Between the "Eyes on the Street" and Urban Safety_](https://github.com/timovanasten/natural-surveillance/blob/main/doc/Measuring%20Natural%20Surveillance%20at%20Scale.pdf).
See the document for a detailed descripton of the project and the obained results.

![front-cover](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/front-cover.png?raw=true)

## Abstract
To create safe urban areas, it is important to gain insight into what influences the (perceived) safety of our cities and human settlements. One of the factors that can contribute to safety is the way urban spaces are designed. Previous work has highlighted the importance of _natural surveillance_: a type of surveillance that is a byproduct of how citizens normally and routinely use the environment. However, studying this concept is not a trivial task. Manual approaches such as observation studies are costly and time-consuming and have therefore often limited themselves to smaller geographical areas. 

In this work, we present a methodology that can automatically provide an estimate of natural surveillance by detecting building openings (i.e. windows and doors) in street level imagery and localizing them in 3 dimensions. The proposed method is able to estimate natural surveillance at the street segment level, while simultaneously being able to gather data on a whole city in a matter of hours. We then apply our method to the city of Amsterdam to analyze the relationship between natural surveillance and urban safety using the Amsterdam Safety Index. 

We conclude that our chosen operationalization of natural surveillance (road surveillability and occupant surveillability) is correlated with decreases in high impact crime and nuisance as well as increases in perceived safety. Furthermore we provide evidence for the existence of a threshold after which extra natural surveillance is no longer associated with higher degrees of safety.

## How to Use
This project is written and tested in Python 3.9.
### 1. Clone the repo and install dependencies
Clone the repository:
```sh
git clone https://github.com/timovanasten/natural-surveillance.git
cd natural-surveillance
```

Optionally, create and activate a virtual environment:

For MacOS/Linux:
```sh
python3 -m venv ./venv
source ./venv/bin/activate
```
or for Windows:
```sh
python3 -m venv ./venv
.\venv\Scripts\activate.bat
```
Lastly, install the dependencies:
```sh
pip install -r requirements.txt
```

### 2. Download the facade labeling model
The trained model used for detecting building openings within the street view imagery can be downloaded [here]( https://drive.google.com/drive/folders/1TfeIcQ8KlEvP1-ewGcTaj3SqU_IpoLUv). 
Next, create the ```/labeling/heatmap_fusion/model``` directory. For example by running
```sh
mkdir ./labeling/heatmap_fusion/model
```
and place ```resnet18_model_latest.pth.tar``` into the ```natural-surveillance/labeling/heatmap_fusion/model``` directory.

Alternatively, point to the path the model is located in ```settings.yaml```.
### 3. Google Cloud API key
Obtain an API key for the Google Street View Static API within the [Google Cloud platform](https://developers.google.com/maps/documentation/streetview/get-api-key) and add it to ```.secrets.yaml```. 
To not overshoot available credit, enter your credit in dollar in ```data/gsv_budget.txt```.

## Datasets
![dataset](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/neighborhood-level.png?raw=true)
A dataset in CSV format with the estimated natural surveillance scores for over 6500 street segments spread over 43 neighborhoods in Amsterdam can be found in the ```data/results``` directory.
To create a nice visualization of the dataset, such as the one above, check out [kepler.gl](https://kepler.gl/demo).

A dataset containing the estimated geolocation of 872 360 building openings is also available. Please contact me for access.

## Running the Code
The code that was ran for the experiment outlined in the thesis document can be found in ```main.py```. 
To obtain your own data, update the EPSG of the local coordinate reference system in `settings.yaml` to one of the area of interest and create a ```Pipeline``` object using either a polygon, address or neighborhood name*:

```python
from shapely.geometry import Polygon
from pipeline import Pipeline, PipelineConfig, PipelineStep

# Option 1: using area polygon
polygon = Polygon([4.4359826, 52.2269100],
                  ...,
                  [4.4364993, 52.2271558])

config = PipelineConfig(pipeline_name, 
                        field_of_view, 
                        sightline_distance, 
                        viewing_angle, 
                        polygon=polygon)

# Option 2: using address
config = PipelineConfig(pipeline_name, 
                        field_of_view, 
                        sightline_distance, 
                        viewing_angle, 
                        address="Mekelweg 4, Delft",
                        distance_around_address=100)

# Option 3: using neighborhood name
config = PipelineConfig(pipeline_name, 
                        field_of_view, 
                        sightline_distance, 
                        viewing_angle, 
                        neighborhood_name="Staatliedenbuurt")

# Create the pipeline
pipeline = Pipeline(config)

# Execute all steps in the pipeline:
pipeline.execute_all()
# ...or only certain step(s):
pipeline.execute_step(PipelineStep.LOCALIZE_OPENINGS)
pipeline.execute_up_to_and_including(PipelineStep.LOCALIZE_OPENINGS)
pipeline.execute_from(PipelineStep.CALCULATE_SIGHTLINES)

# Pipelines are saved after each step. 
# To load a prevously saved pipeline into memory use:
pipeline = Pipeline.load_from_state_file(pipeline_name)
```

_*This repository only contains the neighborhood geometries for Amsterdam. To use other neighborhood boundaries, 
alter `neighborhoods.py` and `settings.yaml`_.
## Contact
For any questions or requests, feel free to contact me at timovanasten@gmail.com.


## Visualizations

![data-collection](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/data-collection.png?raw=true)
![opening-localization](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/opening-localization-overview.png?raw=true)
![opening-detection](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/opening-detection.png?raw=true)
![localization-output](https://github.com/timovanasten/natural-surveillance/blob/main/doc/img/localization-output.png?raw=true)
