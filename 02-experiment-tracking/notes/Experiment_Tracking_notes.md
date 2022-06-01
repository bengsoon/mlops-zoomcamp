## Experiment Tracking Notes

## Video 1: Introduction

### Important concepts:
1. ML Experiment: 
    - the whole process of building an ML model
    - different from A/B testing
2. Experiment run: 
    - each trial in an ML experiment
    - each trial in the modeling (changes in architecture, hyperparameters)
3. A run artifact:
    - Any saved file associated to that experiment run
4. Experiment metadata:
    - All information related to the experiment run

### What is **Experiment Tracking**?
The process of **keeping track** of all the ***relevant information*** from an ML experiment, which includes:
- Source code
- Environment
- Data (different versions)
- Model (different architectures)
- Hyperparameters
- Metrics
- ...

A lot of time it is difficult to know ahead of time what metadata you'd need, but we usually include the standard information.

### Why is Experiment Tracking so important?
1. Reproducibility
2. Organization
3. Optimization

### Why not Spreadsheets?
1. Error prone
2. No standard format
3. Visibility and Collaboration

### MLflow

"An open source platform for the ML lifecycle"

- It's a simple Python package that can be installed with `pip`
- Contains four main modules:
    1. Tracking
    2. Models
    3. Model Registry
    4. Projects (not part of the scope of this Zoomcamp)

### Tracking experiments with MLflow
The MLFlow Tracking module allows you to organize your experiments into runs and to keep track of:
1. Parameters
2. Metrics
3. Metadata
4. Artifacts
5. Models

MLflow also automatically logs extra information about the run:
1. Source code
2. Version of the code (git commit)
3. Start and end time
4. Author

### MLflow in action

To install MLflow:

``` shell
pip install mlflow
```

MLflow comes with a cli:

```
$ mlflow
```


```
Usage: mlflow [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  artifacts    Upload, list, and download artifacts from an MLflow...
  azureml      Serve models on Azure ML.
  db           Commands for managing an MLflow tracking database.
  deployments  Deploy MLflow models to custom targets.
  experiments  Manage experiments.
  gc           Permanently delete runs in the `deleted` lifecycle stage.
  models       Deploy MLflow models locally.
  run          Run an MLflow project from the given URI.
  runs         Manage runs.
  sagemaker    Serve models on SageMaker.
  server       Run the MLflow tracking server.
  ui           Launch the MLflow tracking UI for local viewing of run...
```

Launching MLflow ui will launch a gunicorn server:

```
$ mlflow ui
[2022-05-24 11:47:20 +****] [14391] [INFO] Starting gunicorn 20.1.0
[2022-05-24 11:47:20 +****] [14391] [INFO] Listening at: http://127.0.0.1:5000 (14391)
[2022-05-24 11:47:20 +****] [14391] [INFO] Using worker: sync
[2022-05-24 11:47:20 +****] [14392] [INFO] Booting worker with pid: 14392
```

We then go to the browser to view the page http://127.0.0.1:5000

!["MLflow UI: Create New Experiment"](./img/01_mlflow_UI_create_experiment.png)

MLflow UI: Create New Experiment

![](./img/02_mlflow_UI_exp_name.png)

- Experiment Name - Name of the Experiment
- Artifact Location - Location where you'd save the files for the experiment runs
    - Could be pickle file, local folder etc
    - Could even be an S3 Bucket

![](./img/03_mlflow_UI_experiments.png)

- Important features on the Experiments page

![](./img/04_mlflow_UI_model_registry_error.png)

The reason that we can't view the models registry is because we will need to have a backend database such as postgresql, mysql, sqlite, mssql

## Video 2: Getting Started with MLflow
- Prepare local env
- Install MLflow client and configure a backend
- Add MLflow to the existing notebook, log the predictions and view it on MLflow UI

### Create the environment

First, we should create a conda environment (we'll name it `exp-tracking-env`) so that we do not disrupt our system's installation:

```
conda create -n exp-tracking-env python=3.9
```

Then, we will activate the environment

```
conda activate exp-tracking-env
```

#### Install the required packages

Cristian has already prepared the `requirements.txt` for this section:

``` shell
$ cat requirements.txt

mlflow
jupyter
scikit-learn
pandas
seaborn
hyperopt
xgboost
fastparquet
```

To install these packages, we will point `pip` to our `requirements.txt` within our newly-created conda environment:

```
(exp-tracking-env) $ pip install -r requirements.txt
```

### Starting MLflow with sqlite backend

In the first video, we managed to start the MLflow UI by `mlflow ui`. However, MLflow does not automatically connect itself to the backend database server. Because of this, we were not able to access the model registry function in the UI (see Video 1: Introduction section).

To start MLflow UI with the backend connected to an sqlite database, we have to use the following command (take note of the _triple_ front-slashes):

`(exp-tracking-env) $ mlflow ui --backend-store-uri sqlite:///mlflow.db`

What this command does is that it tells mlflow ui to connect to a backend sqlite database `sqlite:///mlflow.db `.

So, as before, this command will spin up a gunicorn server:

``` shell
2022/05/25 08:21:53 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2022/05/25 08:21:53 INFO mlflow.store.db.utils: Updating database tables
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 451aebb31d03, add metric step
INFO  [alembic.runtime.migration] Running upgrade 451aebb31d03 -> 90e64c465722, migrate user column to tags
INFO  [alembic.runtime.migration] Running upgrade 90e64c465722 -> 181f10493468, allow nulls for metric values
INFO  [alembic.runtime.migration] Running upgrade 181f10493468 -> df50e92ffc5e, Add Experiment Tags Table
INFO  [alembic.runtime.migration] Running upgrade df50e92ffc5e -> 7ac759974ad8, Update run tags with larger limit
INFO  [alembic.runtime.migration] Running upgrade 7ac759974ad8 -> 89d4b8295536, create latest metrics table
INFO  [89d4b8295536_create_latest_metrics_table_py] Migration complete!
INFO  [alembic.runtime.migration] Running upgrade 89d4b8295536 -> 2b4d017a5e9b, add model registry tables to db
INFO  [2b4d017a5e9b_add_model_registry_tables_to_db_py] Adding registered_models and model_versions tables to database.
INFO  [2b4d017a5e9b_add_model_registry_tables_to_db_py] Migration complete!
INFO  [alembic.runtime.migration] Running upgrade 2b4d017a5e9b -> cfd24bdc0731, Update run status constraint with killed
INFO  [alembic.runtime.migration] Running upgrade cfd24bdc0731 -> 0a8213491aaa, drop_duplicate_killed_constraint
INFO  [alembic.runtime.migration] Running upgrade 0a8213491aaa -> 728d730b5ebd, add registered model tags table
INFO  [alembic.runtime.migration] Running upgrade 728d730b5ebd -> 27a6a02d2cf1, add model version tags table
INFO  [alembic.runtime.migration] Running upgrade 27a6a02d2cf1 -> 84291f40a231, add run_link to model_version
INFO  [alembic.runtime.migration] Running upgrade 84291f40a231 -> a8c4a736bde6, allow nulls for run_id
INFO  [alembic.runtime.migration] Running upgrade a8c4a736bde6 -> 39d1c3be5f05, add_is_nan_constraint_for_metrics_tables_if_necessary
INFO  [alembic.runtime.migration] Running upgrade 39d1c3be5f05 -> c48cb773bb87, reset_default_value_for_is_nan_in_metrics_table_for_mysql
INFO  [alembic.runtime.migration] Running upgrade c48cb773bb87 -> bd07f7e963c5, create index on run_uuid
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
[2022-05-25 08:21:54 +0800] [19279] [INFO] Starting gunicorn 20.1.0
[2022-05-25 08:21:54 +0800] [19279] [INFO] Listening at: http://127.0.0.1:5000 (19279)
[2022-05-25 08:21:54 +0800] [19279] [INFO] Using worker: sync
[2022-05-25 08:21:54 +0800] [19280] [INFO] Booting worker with pid: 19280
```

Just like in Video 1, we can access the UI on the browser via http://127.0.0.1:5000

At this point, since we have just started this, we won't have any experiment runs yet.

![](./img/05_mlflow_UI_empty.png)

However, this time, we will be able to access the "Models" tab (Model Registry) with no error:
![](./img/06_mlflow_UI_model_registry_noprob.png)

### Add MLflow to Existing Notebook

#### Copy Code from Module 1

We can now bring in the code from Module 1 (either `duration-prediction.ipynb` or our homework solution) as well as the data

In my case, I copied my version of the homework solution:

<center>

![](./img/07_copy_module1_to_folder.png)
</center>


We also need to create a `models` folder, where we will save all the model artifacts, otherwise we will see an error.

<center>

![](./img/08_create_models_folder.png)

</center>

#### Python Notebook with MLflow

Let's open up our Python notebook (in my case it would be `homework1_solutions.ipynb`).

First we have to ensure that the required Python libraries can be imported and that the Python version is the same as when we previously trained the model.

Then we need to import the MLflow library:

``` python
    import mlflow
```


We also need to set tracking URI to point the library to the SQLite backend database for MLFlow. The URI is the one parameter in the mlflow ui cli:

<center>

![](./img/09_tracking_uri.png)
</center>

``` Python
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

We also need to set the experiment. If the experiment does not exist, MLflow will automatically create the experiment for us.

``` python
    mlflow.set_experiment("nyc_taxi_experiment")
```

```
        2022/05/25 11:20:29 INFO mlflow.tracking.fluent: Experiment with name 'nyc-taxi-experiment' does not exist. Creating a new experiment.
        <Experiment: artifact_location='./mlruns/2', experiment_id='2', lifecycle_stage='active', name='nyc-taxi-experiment', tags={}
```

![](./img/10_mlflow_experiment_created.png)

You can see that the `nyc_taxi_experiment` has been created on MLflow UI

To start tracking our runs, need to append a `with mlflow.start_run()` on our training cell. 

``` python

# start logging with mlflow
with mlflow.start_run():
    ...
```

We will first log a tag called "developer"
``` python
    # set tag for mlflow
    mlflow.set_tag("developer", "Bengsoon")
```

We can then start logging the parameters as we wish. In our case, we are going to save the source of the data for both training and validation:

``` python
    # set data path param
    mlflow.log_param("train-data-path", "./data/fhv_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/fhv_tripdata_2021-02.parquet")
```

Let's say we are going to train a Lasso() model, and for this run we will set our hyperparameter `alpha` = 0.01. We should log this as a parameter in mlflow:
``` python
    # set hyper parameter for Lasso
    alpha = 0.001
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha = alpha)
```

Once we have trained the model, we will calculate its `rmse` score against our validation set. We should also log this as a parameter:
``` python
    # get RMSE and record on mlflow
    rmse = round(mean_squared_error(y_val, y_pred, squared=False),2)
    print("RMSE for training data:", rmse)
    mlflow.log_metric("rmse", rmse)
```

Here is the whole code block for training the model, validating its performance and recording the run information on mlflow:

``` python
# start logging with mlflow
with mlflow.start_run():
    # set tag for mlflow
    mlflow.set_tag("developer", "Bengsoon")

    # set data path param
    mlflow.log_param("train-data-path", "./data/fhv_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/fhv_tripdata_2021-02.parquet")

    # set hyper parameter for Lasso
    alpha = 0.01
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha = alpha)
    
    # train the model
    lr.fit(X_train, y_train)

    # get the y_pred from X_train
    y_pred = lr.predict(X_val)

    # get RMSE and record on mlflow
    rmse = round(mean_squared_error(y_val, y_pred, squared=False),2)
    print("RMSE for training data:", rmse)
    mlflow.log_metric("rmse", rmse)
```

If we go to mlflow UI, we should see that the our new run has been recorded (click Refresh if it did not show up). Note that all the parameters that we have recorded on our notebook as been recorded in MLflow.

![](./img/12_mlflow_new_run.png)



> Notice that the "Source" in MLflow UI shows that it is ipykernel_launcher.py, which needs an additional hack to be done as MLflow is unable to detect the version of the code from ipynb.

To view the run details, we can click on the Start Time value:

![](./img/13_mlflow_click_details.png)

![](./img/14_mlflow_detailed_info.png)



> Notice that the rmse is only a single value. If you are running a model with epochs, you can see the rmse performance over time.

## Video 3: Experiment tracking with MLflow
We are going to try out an [`xgboost` model](https://xgboost.readthedocs.io/en/stable/) and optimize the hyperparameter tuning with a package called [`hyperopt`](https://hyperopt.github.io/hyperopt).

The code is provided by Cristian.

#### Hyperparameter Optimization Tuning

``` python
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)
```

>-  `fmin`: tries to minimize objective function
>-  `tpe`: algorithm that controls the flow
>-  `hp`: hyperparameter space
>-  `STATUS_OK`: signal if the optimization is succesful at the end of each run
>-  `Trials`: will keep track of information from each run

We create an `objective` function that trains the `xgboost` model with a set of hyperparameters (from `hyperopt`) and then validated against our validation data. 
For each set of hyperparameters and the model's corresponding performance score, we record them in `mlflow` by wrapping it around the function.

``` python
def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}

```

We then create the search space dictionary for the XGboost hyperparameters. We use `hp` to create different kinds of statistical distributions for our parameters:

``` python
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}
```



Run the `hyperopt` optimization with `fmin` method (minimize loss)

``` python
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
```


```
Output exceeds the size limit. Open the full output data in a text editor
[17:44:13] WARNING: ../src/objective/regression_obj.cu:203: reg:linear is now deprecated in favor of reg:squarederror.
[0]	validation-rmse:20.79160                          
[1]	validation-rmse:18.41274                          
[2]	validation-rmse:17.00143                          
[3]	validation-rmse:15.83355                          
[4]	validation-rmse:15.25739                          
[5]	validation-rmse:14.79529                          
[6]	validation-rmse:14.61999                          
[7]	validation-rmse:14.51805                          
[8]	validation-rmse:14.46920                          
[9]	validation-rmse:14.44403                          
[10]	validation-rmse:14.42998                         
[11]	validation-rmse:14.43086                         
[12]	validation-rmse:14.43444                         
[13]	validation-rmse:14.44042   
```

MLflow will track all the runs for the hyperopt experiments that we have conducted and we can view the results that we have logged within our `objective` function.
![](./img/16_mlflow_saved_runs.png)


As we have also set the MLflow tag to "model: xgboost" in our function, we can also filter it in MLflow UI:

![](./img/17_mlflow_tags.png)

We can also visually compare them in MLflow UI by selecting all the filtered runs and click compare:

![](./img/18_mlflow_compare.png)


#### Hyperparameter Tuning Visualization

#### Parallel Coordinates Plot 

Parallel Coordinates Plot allows us to see how different combinations of the hyperparameters (from our runs) affect our metric (RMSE). In our case, we want to minimize RMSE, so we can even visually select the runs that provide the best performance:

![](./img/19_mlflow_Parallel_coordinates_plot.png)

#### Scatter Plot

We can also visualize the different hyperparameters against our RMSE metric...In our case, we can see that `min_child_weight` has some correlation with our metric:

![](./img/20_mlflow_scatter_plot.png)

#### Contour Plot

The contour plot provides us a visualization on the effects of two variables against RMSE:

![](./img/21_mlflow_contour_plot.png)

#### Model Selection

There is no hard-and-fast rule to model selection, but here we will consider the following:
1. Best metric performance (ie lowest RMSE)
2. Lowest training time: sometimes we may have runs with RMSE scores not too far off from each other, but the training time taken is significantly shorter. The simpler the better, if possible.

To do so, we can go back to the Experiments tab in the MLflow UI to select the model that fulfils our criteria:
1. We know that we will use xgb model, so we will leave the tag filter from before.
2. We then sort the model by ascending RMSE values.
3. Select the model that fulfils our criteria

![](./img/22_mlflow_model_selection.png)

Selecting the experiment run above, we can look at the hyperparameter values used in this run and rerun the training as our selected model in our notebook.

![](./img/23_mlflow_selected_run.png)


``` python
# Hyperparameter for run 09923bbad64045ca837a1656254ce756

search_space = {
    'max_depth': 4,
    'learning_rate': 0.14493221791716185,
    'reg_alpha': 0.012153110171030913,
    'reg_lambda': 0.017881159785939696,
    'min_child_weight': 0.674864917045824,
    'objective': 'reg:linear',
    'seed': 42
}
```

Instead of wrapping our code with `with mlflow.start_run()`, we can use mlflow [`autolog`](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging) for xgboost:
``` python
mlflow.xgboost.autolog()

booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
```

`autolog` will automatically log the common parameters, metrics and artifacts for the particular model that we use (in this case, XGBoost). 

![](./img/24_mlflow_autolog.png)

#### The Artifacts

It also automatically saves the artifacts of the model as well as its corresponding environments / requirements for its usability in production phase. For `xgboost`, it also saves `feature_importance_weight.json` by default.

![](./img/25_mlflow_autolog_artifacts.png)


Under the MLmodel, it provides us information/metadata regarding the model that was saved. It also shows that the model can be used as a python_function (`pyfunc`) or as an XGBoost model. Here, we can also download the model binary.

![](./img/25_mlflow_autolog_MLmodel.png)

At the top level of the Artifacts, it also provides us information on how we can use the saved models to make predictions (as inference models):

![](./img/26_mlflow_artifacts.png)

In our `mlflow_notebook_for_notes.ipynb`, we can copy the code above for "Pandas DataFrame" to use it to test out the inference mode of the model:
- We first have to make a reference to the saved model with the `runs:/{run_id}/model`, which is a path readable by MLflow
- We can load the model as PyFuncModel or as XGBoost model (in our case we will load it as XGBoost Model)


``` python
logged_model = 'runs:/01d97a61959f42ba964175e922ee9573/model'
```

#### Load model as `PyFuncModel`

To load as a PyFuncModel, we can use the method `mlflow.pyfunc.load_model()`

``` python
   # Load model as a PyFuncModel.
   loaded_model = mlflow.pyfunc.load_model(logged_model)
   loaded_model

   >> mlflow.pyfunc.loaded_model:
      artifact_path: model
      flavor: mlflow.xgboost
      run_id: 01d97a61959f42ba964175e922ee9573
```

Remember that we already have assigned variables for our validation set features?

``` python
    # preprocessed DataFrame with DictionaryVectorizer
    X_val = dv.transform(X_val_dict)

    # convert X_val into DMatrix type
    valid = xgb.DMatrix(X_val, label=y_val)
```

As shown above in the MLflow UI, PyFuncModel allows us to use Pandas DataFrame features for its prediction method, unlike XGBoost models which requires us to convert features into `DMatrix` type.

Passing in `valid` into `loaded_model.predict()` will return an `TypeError: Not supported type for data.<class 'xgboost.core.DMatrix'>`



``` python
    loaded_model.predict(valid)

    >> ............
    >> TypeError: Not supported type for data.<class 'xgboost.core.DMatrix'>
```

However, the model will be able to make prediction by passing in our preprocessed df `X_val`:

``` python
    loaded_model.predict(X_val)
    >> array([20.216328, 20.216328, 20.216328, ..., 20.216328, 20.216328,
       20.216328], dtype=float32)
```

#### Load Model as `XGBoost` Model

On the flip side, we can also load the model as `XGBoost` model:

``` python
    # Load model as XGBmodel
    xgb_model = mlflow.xgboost.load_model(logged_model)
    xgb_model

    >> <xgboost.core.Booster at 0x7ffac359c070>
   
```

We can see that it is a fully-functional XGBoost model with its corresponding methods:

![](./img/27_xgboost_model.png)

Since it is an XGBoost model, we will need to use the DMatrix-typed feature inputs for our validation set:

``` python
    xgb_model.predict(valid)
    >>> array([20.216328, 20.216328, 20.216328, ..., 20.216328, 20.216328,
       20.216328], dtype=float32)
```


#### Model Environments

MLflow also provides the Python environment from which the model was trained on. This will help the deployment of the model in the production stage to replicate the same environment as the training stage.

MLflow automatically provides 3 types of environment files:
- `conda.yaml` for Conda
- `requirements.txt` for pip
- `python_env.yaml` for virtualenv

![](./img/28_mlflow_autolog_envs.png)

## Video 4: Model Management 

This is a general machine learning lifecycle:

![](./img/29_ml_lifecycle_neptune.png)

_Source: https://neptune.ai/blog/ml-experiment-tracking_

- Here we can see that Experiment Tracking is just a small subset of the whole lifecycle
- Model Management is an overarching process that also includes Model Versioning, Model Deployment and Scaling Hardware on top of Experiment Tracking

### What's wrong with Folder-based model management?
![](./img/30_whats_wrong_with_folders.png)
- Error prone: we can easily replace older models
- No versioning
- No lineage: no information on which data version that the model is derived from or the hyperparameters that it uses etc o

### Model Management with MLflow

Adapting from our previous XGBoost model, we will save the artifact into MLflow and log the parameters manually. We will also log the DictVectorizer preprocessor as an artifact

First, we have to turn off the `autolog` to avoid logging the model twice:

``` python 
    # Turn off autolog
    mlflow.xgboost.autolog(disable=True)
```

We then add the following to the code:

1. To log our custom parameters, we can pass in our `params` dictionary directly into `mlflow.log_params()`:

   ``` py
    # Hyperparameter for run 09923bbad64045ca837a1656254ce756
    params = {
        'max_depth': 4,
        'learning_rate': 0.14493221791716185,
        'reg_alpha': 0.012153110171030913,
        'reg_lambda': 0.017881159785939696,
        'min_child_weight': 0.674864917045824,
        'objective': 'reg:linear',
        'seed': 42
    }
    mlflow.log_params(params)
   ```
2. To log metric "rmse":

    ``` py
    # adapted from MLflow with LR
    mlflow.log_metric("rmse", rmse)
    ```

3. To log xgboost model to with `mlflow.xgboost.log_model` to ensure that we are saving xgboost model:
    ``` py
    # log xgboost model to mlflow
    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    ```

4. To log the preprocessor `DictVectorizer` that we created (and exported) at first:
    ``` py
    # log the preprocessor DictVectorizer
    with open("models/preprocessor.bin", "wb") as f_out:
        pickle.dump(dv, f_out)
    
    mlflow.log_artifact("models/preprocessor.bin", artifact_path="preprocessor")
    ```

In its totality, here's how we can train an XGBoost model and log it with our custom params, metrics and artifacts:

``` py
    # Adapting from our previous XGBoost model, we will save the artifact into MLflow and log the parameters manually
    # We will also log the DictVectorizer preprocessor as an artifact

    # Turn off autolog
    mlflow.xgboost.autolog(disable=True)


    with mlflow.start_run():
        # Hyperparameter for run 09923bbad64045ca837a1656254ce756
        params = {
            'max_depth': 4,
            'learning_rate': 0.14493221791716185,
            'reg_alpha': 0.012153110171030913,
            'reg_lambda': 0.017881159785939696,
            'min_child_weight': 0.674864917045824,
            'objective': 'reg:linear',
            'seed': 42
        }

        mlflow.log_params(params)
        

        booster = xgb.train(
                    params=params,
                    dtrain=train,
                    num_boost_round=1000,
                    evals=[(valid, 'validation')],
                    early_stopping_rounds=50
                )

        # get the y_pred from X_train
        y_pred = booster.predict(valid)

        # get RMSE and record on mlflow
        rmse = round(mean_squared_error(y_val, y_pred, squared=False),2)
        print("RMSE for training data:", rmse)
        mlflow.log_metric("rmse", rmse)

        # log xgboost model to mlflow
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        # log the preprocessor DictVectorizer
        with open("models/preprocessor.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
        
        mlflow.log_artifact("models/preprocessor.bin", artifact_path="preprocessor")

```

We can see that our run has been saved, along with the parameters and metrics that we have manually logged. Under the artifacts, we have `models_mlflow` and `preprocessor` that were saved:

![](./img/31_mlflow_xgboost_manual.png)


#### Load XGBoost Model

Once the model has been successfully trained, we can load the model from MLflow in our notebook, just as before (see notes for Video 3):

``` python
    logged_model = 'runs:/237dc915805441e8bfe958044ede7b18/models_mlflow'

    # Load model as a xgboost.
    xgb_model = mlflow.xgboost.load_model(logged_model)

    # Prediction
    y_valid = xgb_model.predict(valid) # using DMatrix-typed validation data

    y_valid
    >>> array([20.216328, 20.216328, 20.216328, ..., 20.216328, 20.216328,
       20.216328], dtype=float32)

```

## Video 5:

![](./img/32_model_registry.png)

Source: https://neptune.ai/blog/model-registry-makes-mlops-work

Model Registry does not perform the deployment of the model, but it merely is a registry where we label the versions and the stages of the model. To deploy the model, we will require to implement CI/CD to communicate with Model Registry.

As Data Scientists, our job is not to deploy the model, but to decide which model will be used for production.

A few common considerations to take account (apart from performance metrics - RMSE):
- Duration
- Size of the model

In my notebook, I have run several runs with other model architectures such as [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) and [`ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) along with previous runs with `XGBoostRegressor` and `Lasso`

Let's assume that these are the models that we intend to stage, we sort the runs by ascending RMSE scores:

![](./img/33_selecting_runs_for_staging.png)

But as mentioned above, we also usually need to also take note of the training duration as that is indicative of the model sizes. However, in my example, even though `GradientBoostingRegressor` architecture takes the longest to train, its serialized model is only a mere 108.47KB:

![](./img/34_gbr_filesize.png)
Serialized Model File Size for `GradientBoostingRegressor` run



***Registering Model*** 

Considering their (slightly) lower RMSE scores, we will register all the runs to the Model Registry. To do so, for each individual run, we click on the Register Model button:

![](./img/35_register_gbr.png)


As we have not registered a model before, we will need to Create New Model (we will name it `nyctaxi_tripduration_regressor`):

![](./img/36_create_new_model.png)

- Side Note: in case if it confuses you as much as it did with me, even though we use different model architectures for the runs in our Experiment, we do not call them "models". They remain as runs for the experiment that we are doing, whose goal is to ultimately build a single model of NYC Taxi Trip Duration Regressor. It gets clearer in these next few steps where we start to register the "runs" as the "production model" for the NYC Taxi Trip Duration Regressor that we are building.


Once we have registered all the models, we can see that the icons under the "Models" column (back in the "Experiments" tab) for the runs have changed:

![](./img/37_models_tab.png)

If we click on the "Models" tab, we can also see that our model has been registered with the "Latest Version" being Version 4:

![](./img/38_registered_model.png)

If we click on the `nyctaxi_tripduration_regressor` model, we can add some information in the "Description":

![](./img/39_model_description.png)


Here we can see the different versions of the model. The versions are automatically numbered by MLflow based on the sequence of when we registered the runs:

![](./img/40_model_versions.png)


Also note that MLflow does not automatically inherit the metadata (like model architecture, run tags etc) from the runs, so we have to label them manually. We can do so by clicking on the "Source Run" link, which will get us back to the run from which the version is referring to. For example, for Version 2, we can see that it refers to the XGBoost Run that we registered first:

![](./img/41_version_2.png)

_Version of the model and its Source Run_

![](./img/42_xgboost_is_v2.png)

_Corresponding run of Model Version 2 is the XGBoost run_

We will add the model architecture in the "Description" field for each of the version:

![](./img/43_v2_add_description.png)

### Transitioning Model Versions

Back in the main page of the Models tab, we can transition each of these Versions to different stages either using MLflow UI or MLflow API in Python. 

For now let's transition Version 1 to "Production" stage while the other remaining versions will reside in "Staging" using MLflow UI. To do so, we click on each of the version and select the stage to transition it into. Below is the example for Version 1:

![](./img/44_transition_v1_production.png)

A prompt will show up for confirmation with an option to "Transition existing Production model versions to Archived". For now, we have no models in Production, so that does nothing for us:

![](./img/45_transition_v1_archive.png)

Once we have done that for the others, we can see on the main screen that we have all four versions have their own corresponding stages:

![](./img/46_staged_versions.png)

#### Using MLflow Client

We will use `MLflowClient` in Python in order to communicate with our model

``` py
    from mlflow.tracking import MLflowClient
```

We need to instantiate the `Client`
``` py
    # instantiate the client
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
```

With the `Client`, we can create experiments

``` py
    # we can create experiments
    client.create_experiment(name="test")
```

To list the current experiments,
``` py
    # list experiments
    client.list_experiments()

>>> [<Experiment: artifact_location='./mlruns/2', experiment_id='2' lifecycle_stage='active', name='nyc-taxi-experiment', tags={}>, 
<Experiment: artifact_location='./mlruns/4', experiment_id='4', lifecycle_stage='active', name='test', tags={}>]

#### Get the runs in `nyc-taxi-experiment`

``` py
    from mlflow.entities import ViewType

    runs = client.search_runs(
        experiment_ids= "2",
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"]
    )
```

- `search_runs` is a simplified version of SQL `WHERE` clause.
- We first need to specify which experiment id that we are referring to (`nyc_taxi_experiment`'s id is 2).
- The `filter_string` allows us to filter the runs.
- The `run_view_type` value `ViewType.ACTIVE_ONLY` shows only the active runs (and not deleted runs)
- `max_results` showing only 5 results
- `order_by` - ordering the results by `metrics.rmse` and `ASC` ascending (like SQL)

The `runs` will spit out a myriad of information from the experiments, so we will just extract the ones we want:

``` py
        # the runs have a ton of information, but we can choose to only view the run_id and rmse
        for run in runs:
            print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']}")

        >>> run id: 572b0ed57e3748ac965af939c28f3d0e, rmse: 14.434830071276503
            run id: be2c1e01147c422fa40f9aca18025447, rmse: 14.436014044219164
            run id: ebb70531b3d348808e9c13bd801f335a, rmse: 14.436321257721108        
            run id: 494b84cae6e84edb8e3caa51603c2318, rmse: 14.436641761102681        
            run id: 1e835cd7d1544c01ac90901218a8757e, rmse: 14.44200357693116
```

#### Registering a New Model

Say we want to register the run `572b0ed57e3748ac965af939c28f3d0e` in the Model Registry.

We need to import mlflow and set it up (if we had not done so)

``` py
    # import and set up mlflow
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```

``` py
    # set up variables
    RUN_ID = "572b0ed57e3748ac965af939c28f3d0e"
    MODEL_URI = f"runs:/{RUN_ID}/model"
```

But first let's ensure that the run has not been registered:

``` py
    assert client.search_model_versions(f"run_id = '{RUN_ID}'") == [], "Run has been registered!"
```

Then we can register the model with the `MODEL_URI` (which is linked to the RUN ID and saved model)

``` py
    mlflow.register_model(model_uri = MODEL_URI, name="nyctaxi_tripduration_regressor")

    >>> Registered model 'nyctaxi_tripduration_regressor' already exists. Creating a new version of this model...
2022/06/01 14:51:46 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: nyctaxi_tripduration_regressor, version 5
Created version '5' of model 'nyctaxi_tripduration_regressor'.
<ModelVersion: creation_timestamp=1654066306963, current_stage='None', description=None, last_updated_timestamp=1654066306963, name='nyctaxi_tripduration_regressor', run_id='572b0ed57e3748ac965af939c28f3d0e', run_link=None, source='./mlruns/2/572b0ed57e3748ac965af939c28f3d0e/artifacts/model', status='READY', status_message=None, tags={}, user_id=None, version=5>
```

#### Transitioning the newly registered model 

We will now transition the model to staging. Before we do that, let's find the latest versions

``` py 
    # get the latest versions
    model_name = "nyctaxi_tripduration_regressor"
    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        print(f"Version: {version.version}, Stage: {version.current_stage}")

    >> Version: 1, Stage: Production
       Version: 4, Stage: Staging
       Version: 5, Stage: None
```

Let's transition version 5 to "Staging"

``` py
    # transition version 5 to "Staging"
    stage = "Staging"
    version = 5

    client.transition_model_version_stage(
        name = model_name,
        version = version,
        stage = stage,
        archive_existing_versions = False
    )

    >> <ModelVersion: creation_timestamp=1654066306963, current_stage='Staging', description=None, last_updated_timestamp=1654072597722, name='nyctaxi_tripduration_regressor', run_id='572b0ed57e3748ac965af939c28f3d0e', run_link=None, source='./mlruns/2/572b0ed57e3748ac965af939c28f3d0e/artifacts/model', status='READY', status_message=None, tags={}, user_id=None, version=5>
```
