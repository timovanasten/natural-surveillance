# “Eyes on the Street”: Estimating Natural Surveillance Along Amsterdam’s City Streets Using Street-Level Imagery
Related research: https://link-springer-com.tudelft.idm.oclc.org/chapter/10.1007/978-3-031-31746-0_12

## Abstract
Neighborhood safety and its perception are important determinants of citizens’ health and well-being. Contemporary urban design guidelines often advocate urban forms that encourage natural surveillance or “eyes on the street” to promote community safety. However, assessing a neighborhood’s level of natural surveillance is challenging due to its subjective nature and a lack of relevant data. We propose a method for measuring natural surveillance at scale by employing a combination of street-level imagery and computer vision techniques. We detect windows on building facades and calculate sightlines from the street level and surrounding buildings across forty neighborhoods in Amsterdam, the Netherlands. By correlating our measurements with the city’s Safety Index, we also validate how our method can be used as an estimator of neighborhood safety. We show how perceived safety varies with window level and building distance from the street, and we find a non-linear relationship between natural surveillance and (perceived) safety.

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
