```
gcloud config list --format 'value(core.project)' 
gcloud config set project <PROJECT_NAME>
```
```
jupyter nbextension install --py widgetsnbextension --sys-prefix
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

1. [Select or create a GCP project.](https://console.cloud.google.com/cloud-resource-manager)

2. [Make sure that billing is enabled for your project.](https://cloud.google.com/billing/docs/how-to/modify-project)

3. [Enable the AI Platform Training & Prediction and Compute Engine APIs.](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component)

WitWidget support:
- https://github.com/microsoft/vscode-jupyter/wiki/IPyWidget-Support-in-VS-Code-Python


scripts:
    - [DONE] clear

notebook:
    - [DONE] 00 data prepare
    - [DONE] 01 local train launch
    - 02 cloud train
    - [DONE] 03 model setup & deploy
    - [WORKING] 04 inference / explanation / visualization (missing visualization)

module:
    - trainer