```
gcloud config list --format 'value(core.project)' 
gcloud config set project <PROJECT_NAME>
```

1. [Select or create a GCP project.](https://console.cloud.google.com/cloud-resource-manager)

2. [Make sure that billing is enabled for your project.](https://cloud.google.com/billing/docs/how-to/modify-project)

3. [Enable the AI Platform Training & Prediction and Compute Engine APIs.](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component)

WitWidget support:
- https://github.com/microsoft/vscode-jupyter/wiki/IPyWidget-Support-in-VS-Code-Python

# References
- [Introduction to Vertex Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview)
- [Use TensorFlow with Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/tensorflow)
- [AI Explanations: Explaining a tabular data model](https://colab.research.google.com/github/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/explanations/tf2/ai-explanations-tabular.ipynb#scrollTo=mHF9VCProKJN)
- [AI Explanations: Deploying an image model](https://colab.research.google.com/github/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/explanations/tf2/ai-explanations-image.ipynb#scrollTo=mHF9VCProKJN)