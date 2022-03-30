## deploy 24/7 recording to aks
To have a 24/7 recording running in kubernetes and storing the results to azure blob follow these steps
````bash
az login
az acr login --name fkkstudents
````
make sure you are in the ./camera_stream_recording/src folder
````bash
docker build -t fkkstudents.azurecr.io/recording/camera_recorder .  
docker push fkkstudents.azurecr.io/recording/camera_recorder
````

get .env file with the correct credentials (a backup is stored in the SaveNow storage account fkk247/credentials)

````bash
az aks get-credentials --resource-group fkkstudents --name fkkstudents
kubectl apply -f deployament.yaml
````

## Capture Videostream

This tool has two basic mode. At first you can generate training data for manual labeling in cvat [here is the relevant fork](https://github.com/jul095/cvat).

The other mode is for planned capturing the Videostream.

## Requirements
I recommend using a virtual Env or a conda/miniconda Environment for executing this scripts.
Please install all necessary packages with 
```pip3 install -r requirements.txt```

If you want to do it the manual way you need: 
- opencv
- python-decouple
- numpy
- python-dateutil

For the camera access and automatic upload to the azure file storage you need to provide credentials, usernames and tokens
like in this [example file](env.example). Run `cp env.example .env` and place your secrets into `.env`
If you don't want to use the azure upload function, the sas_token and filepath is not necessary for local capturing and storing videos.

## Quick Start Normal Video Capturing

Just run 

```python3 main.py```

and a video will be captured with a live view. To finish the capturing just press `q` on your keyboard. 

## Generating Video Training data

Run 

```python main.py -r -rd 2 -vd 1```
This runs the script 2 minutes and every 1 minute a new video file will be created

## Generate Random Images during runtime
If you just want some random images during the script runtime, just run  

```python3 main.py -i ```
This will save every minute one image from the stream. This is perfect for generating a huge amount of training data for labeling.

## Upload the files direct to azure
Run ```python3 main.py -u``` uploads the video file after capturing finished directly to azure file storage.



