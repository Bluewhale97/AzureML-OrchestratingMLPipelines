## Introduction

In Azure machine learning you can run workloads as experiments. Generally you will want to separate overall processes into individual tasks and orchestrate the tasks as pipelines of connected steps which is profit to engineer your machine learning life cycle. 

The term pipeline has different meanings in different scenarios. In Scikit-Learn, its definition is the assignment of data preprocessing transformations with a traninig algorithm; in Azure DevOps, it is to perform the build and configuration tasks required to deliver software. In this article, we talk more about Azure machine learning pipelines which encapsulate steps that can run as an experiment.

By the way, as the tutorial said, it's perfectly feasible to have an Azure DevOps pipeline with a task that initiates an Azure Machine Learning pipeline, which in turn includes a step that trains a model based on a Scikit-Learn pipeline!

Let's talk about how to create, publish and schedule an Azure machine learning pipeline in Azure machine leanring studio.

## 1. Some definitions of pipelines

In Azure machine learning, a pipeline is a workflow of machine leanring stakss in which each task is implemented as a step.

Tutorial specifies them very exactly: steps can be arranged sequentially or in parallel, enabling you to build sophisticated flow logic to orchestrate machine learning operations. Each step can be run on a specific compute target, making it possible to combine different types of processing as required to achieve an overall goal. A pipeline can be executed as a process by running the pipeline as an experiment. Each step in the pipeline runs on its allocated compute target as part of the overall experiment run. You can publish a pipeline as a REST endpoint, enabling client applications to initiate a pipeline run. You can also define a schedule for a pipeline, and have it run automatically at periodic intervals.

There are many kinds of steps supported by Azure Machine Learning pipelines, each with its own specialized purpose and configuration options. Including:

![image](https://user-images.githubusercontent.com/71245576/116328264-a62f9000-a796-11eb-8abe-ad00f78abf60.png)

## 2. Creating pipelines

To create a pipeline you must define each step and then create a pipeline that includes the step. The configuration of each step depends on the step type. For example, define two PythonScriptStep steps to prepare data and then train a model:

```python
from azureml.pipeline.steps import PythonScriptStep

# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster')

# Step to train a model
step2 = PythonScriptStep(name = 'train model',
                         source_directory = 'scripts',
                         script_name = 'train_model.py',
                         compute_target = 'aml-cluster')
```

After defining the steps, assign them to a pipeline and run it as an experiment:
```python
from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

# Construct the pipeline
train_pipeline = Pipeline(workspace = ws, steps = [step1,step2])

# Create an experiment and run the pipeline
experiment = Experiment(workspace = ws, name = 'training-pipeline')
pipeline_run = experiment.submit(train_pipeline)
```
You need to pass data between pipeline steps: you can use the PipelineData object which is a special kind of DataReference that references a location in a datastore and create a data dependency between pipeline steps.

You can view a PipelineData object as an intermediary store for data that must be passed from a step to a subsequent step.

![image](https://user-images.githubusercontent.com/71245576/116328931-1db1ef00-a798-11eb-92f5-825058d7bdc9.png)

The procedures to use a PipelineData object are:

1. define a PipelineData object that references a location in a datastore like this
```python
# Define a PipelineData object to pass data between steps
data_store = ws.get_default_datastore()
prepped_data = PipelineData('prepped',  datastore=data_store)
```
2. Pass the PipelineData object as a script argument in steps that run scripts like this:
```python
# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         # Script arguments include PipelineData
                         arguments = ['--raw-ds', raw_ds.as_named_input('raw_data'),
                                      '--out_folder', prepped_data],
                         # Specify PipelineData as output
                         outputs=[prepped_data])
```
3. You should specify the PipelineData as inputs or outputs:
```python
# Specify PipelineData as output
                         outputs=[prepped_data])
```

In the scripts themselves you can obtain a reference to the PipelineData object from the script argument and use it like a local folder:
```python
# code in data_prep.py
from azureml.core import Run
import argparse
import os

# Get the experiment run context
run = Run.get_context()

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--raw-ds', type=str, dest='raw_dataset_id')
parser.add_argument('--out_folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

# Get input dataset as dataframe
raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()

# code to prep data (in this case, just select specific columns)
prepped_df = raw_df[['col1', 'col2', 'col3']]

# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'prepped_data.csv')
prepped_df.to_csv(output_path)
```

For reusing your pipeline steps. By default, the step output from a previous pipeline run is reused without rerunning the step provided the script, source directory, and other parameters for the step have not changed. Step reuse can reduce the time it takes to run a pipeline, but it can lead to stale results when changes to downstream data sources have not been accounted for.

You can control reusing for an individual step, like statement under a step:
```python
 # Disable step reuse
 allow_reuse = False)
```                         
Now you can force all steps to run, 
```python
pipeline_run = experiment.submit(train_pipeline, regenerate_outputs=True)
```
regenerate_outputs parameter is to run regardless of individual reuse configuration of steps.


## 3. Publishing a pipeline
You can publish the pipelines to your clients, specifically, publish it to create a REST endpoint through which the pipeline can berun on demand.

Use its publish method to publish a pipeline:
```python
published_pipeline = pipeline.publish(name='training_pipeline',
                                          description='Model training pipeline',
                                          version='1.0')
```
You can call the publish method on a successful run of the pipeline:
```python
# Get the most recent run of the pipeline
pipeline_experiment = ws.experiments.get('training-pipeline')
run = list(pipeline_experiment.get_runs())[0]

# Publish the pipeline from the run
published_pipeline = run.publish_pipeline(name='training_pipeline',
                                          description='Model training pipeline',
                                          version='1.0')
```
Review the pibepine which has been published, you can view it in Azure machine learning studio or determine the URI of its endpoint like this:
```python
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)
```
To yse a published pipeline, you need to initiate it. You need to make an HTTP request to its REST endpoint, passing an authorization header with a token for a service principal with permission to run the pipeline, and a JSON payload specifying the experiment name. The pipeline is run asynchronously, so the response from a successful REST call includes the run ID. You can use this to track the run in Azure Machine Learning studio.

For example, the following Python code makes a REST request to run a pipeline and displays the returned run ID.
```python
import requests

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline"})
run_id = response.json()["Id"]
print(run_id)
```

## 4. Using pipeline parameters

You can increase the flexibility of a pipeline by defining parameters. For example, define parameters for a pipeline using a PipelineParameter object for each parameter:
```python
from azureml.pipeline.core.graph import PipelineParameter

reg_param = PipelineParameter(name='reg_rate', default_value=0.01)
```
Specify them at least one step:
```python
step2 = PythonScriptStep(name = 'train model',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         # Pass parameter as script argument
                         arguments=['--in_folder', prepped_data,
                                    '--reg', reg_param],
                         inputs=[prepped_data])
```                         

Run a pipeline with a parameter:
```python
response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline",
                               "ParameterAssignments": {"reg_rate": 0.1}})
```

## 5. Scheduling pipelines

After you have published a pipeline, you can initiate it on demand through its REST endpoint, or you can have the pipeline run automatically based on a periodic schedule or in response to data updates.

To schedule a pipeline to run at periodic intervals, you must define a ScheduleRecurrence that determines the run frequency, and use it to create a Schedule.

For example, the following code schedules a daily run of a published pipeline.

```python
from azureml.pipeline.core import ScheduleRecurrence, Schedule

daily = ScheduleRecurrence(frequency='Day', interval=1)
pipeline_schedule = Schedule.create(ws, name='Daily Training',
                                        description='trains model every day',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='Training_Pipeline',
                                        recurrence=daily)
```
To schedule a pipeline to run whenever data changes, you must create a Schedule that monitors a specified path on a datastore, like this:

```python
from azureml.core import Datastore
from azureml.pipeline.core import Schedule

training_datastore = Datastore(workspace=ws, name='blob_data')
pipeline_schedule = Schedule.create(ws, name='Reactive Training',
                                    description='trains model on data change',
                                    pipeline_id=published_pipeline_id,
                                    experiment_name='Training_Pipeline',
                                    datastore=training_datastore,
                                    path_on_datastore='data/training')
```                                    

## Reference:
Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/
