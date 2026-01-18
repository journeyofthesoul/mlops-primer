# MLOps Workflow Primer (Local Kubernetes w/ scikit-learn)

This repository is a lightweight, end‑to‑end MLOps / ML‑workflow primer designed to run **entirely on a local machine** using Kubernetes running on Vagrant (VirtualBox). It is a small simulation environment that helps DevOps / SRE engineers learn the blocks of an ML Pipeline and get a introductory primer into what MLOps is. Having it run locally within the confines of one's personal laptop allows for free experimentation without fear of incurring cloud costs and having it run on Kubernetes on VMs gives it a close feel to a DevOps Engineer's playing field and allows for a peak into what infrastructure issues/concerns an MLOps infrastructure might encounter when deployed full-scale - again, without fear of incurring cloud costs.

As such, the goal is **not** model sophistication, but to demonstrate visually the following blocks of an ML Workflow/Pipeline as simple as possible:

Machine Learning Pipeline
* Data ingestion from an external source
* Feature Engineering
* Model Training
* Model Evaluation
* Serving the Model to end users (Inference) via API

MLOps Technology (using k8s and MLFlow)
* Automation of Ingestion, Experimentation, Training and Evaluation using Python + k8s CronJob
* Workflow for Model Promotion, Tagging, Aliasing
* Model Persistence and Registry (allows for manual ops, rollbacks to older versions of the Model)
* Experiment Tracking  

## Service Description

The API predicts whether the price for the SPDR S&P 500 ETF Trust (Ticker = SPY) would go up or down the following day. With a simple API call, you'll get one of two answers - UP or DOWN. Internally, the model is constantly trained on past data. I should stress, that the ML model used is arbitrary, something quick to set up and help the DevOps engineer see what the blocks of an ML Pipeline is. The goal is not model sophistication so the accuracy is pretty low.  

## Setup Procedure

The command below will set up everything - the entire kubernetes cluster, the jobs, services, mlflow, and bring the API up. Boot-up time still needs to be improved, but everything should be ready 15 minutes after (usually shorter).
```bash
vagrant up
```
To call the API serving the model, run the following curl command (with or without the threshold query parameter):
```bash
% curl http://192.168.56.21:30951/predictionForTomorrow 
{"prediction":"DOWN","confidence":0.0,"threshold":0.5,"model_source":"mlflow","model_alias":"champion"}
% curl "http://192.168.56.21:30951/predictionForTomorrow?threshold=0.3"
{"prediction":"DOWN","confidence":0.0,"threshold":0.3,"model_source":"mlflow","model_alias":"champion"}
```
Note: The IP of 192.168.56.21 is fixed, for a one worker node cluster (with one control plane node) which is configured by default in the _Vagrantfile_. Likewise, the port of 30951 had also been configured fixed as the Service NodePort. If you download this repo AsIs and run _vagrant up_, none of these parameters need to be changed.  

To bring the service down:
```bash
vagrant destroy
```
Or, to shut down the infrastructure temporarily (shut down the VMs):
```bash
vagrant halt
```
Either way, _vagrant up_ will bring back the entire infrastructure.
