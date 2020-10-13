# Load forecasting on GitLab CI/CD

This Repository utilizes GitLab's CI/CD to train and evaluate RNN-Models in the cloud either automatically or when triggered by a user.



## Structure

The CI implementation of this Repository features three stages with one job each.

The first stage uses a shell environment to set up a Docker Image the next jobs can be executed on. This Image has all necessary dependancies including the plf-util Python Package already installed and is only built, if its configuration (Dockerfile) or the dependancies change or there is no saved Image (e.g. first run).

The second stage uses the Docker Image created in the first stage and runs the fc_train.py script. The model file is saved as a GitLab Artifact under ./oracles. There are three ways to tell this stage, which station's data to perform the training on (see Usage)

The third stage uses the second stage's job's artifact and runs an evaluation on it. The output is in turn stored as an artifact in ./oracles. Its behaviour is adaptive to the second stage, making sure the evaluation happens for the station the model was trained for.



## Requirements

To be able to  use this CI implementation when forking, there have to be configured runners with a `shell` and a `docker` environment accessible to the Repository. The tags assigned to these runners should also be `shell` and `docker`.



## Usage
There are three ways to interact with the CI implementation:

* By automatically detecting file changes [Removed]
* Using the commit message when pushing to remote
* By using the GitLab API

##### automatically detecting file changes [Removed]
The CI is set up in a way that it automatically detects changes to the config.json files of the nine included Gefcom2017 stations. If one or multiple config.json files are changed and pushed to remote, training and evaluation automatically starts for the corresponding stations. This ignores anything specified within the commit message of the push event, yet only happens when a push event is detected. When triggering a pipeline using GitLab's API, file changes are ignored.

##### using the commit message
If the pipeline is triggered by a push event, but no file changes to any config.json files are detected, the CI tries to extract the necessary path to a station config from the commit message. Similarly to `[ci skip]` or `[skip ci]`, that skips the CI completely if included in the commit message, `[<path to config>]` can be used to specify this path (inside of /targets). If the commit message does not include a set of `[]`, no pipeline will be created (same as using [ci skip]).

##### using the GitLab API
When triggering a pipeline by using GitLab's API, the config path has to be specified as a variable with the name `STATION` when performing the call. Triggering a pipeline using the API technically reruns the previous commit's pipeline, but overrides all other methods of interaction: If the last commit's message is `"changed something [ri]"` and it contains a change to `targets/gefcom2017/mass_data/config.json`, but its pipeline is rerun by an API call with `STATION = gefcom2017/ct_data`, training and evaluation is **only** performed for ct_data. Such an API call could look like this:

```bash
curl -X POST \
     -F token=TOKEN \
     -F "ref=REF_NAME" \
     -F "variables[STATION]=gefcom2017/mass_data" \
     https://your-gitlab-instance.com/api/v4/projects/your-project-id/trigger/pipeline
```
The Token can be aquired under `Settings/CI_CD/Pipeline_Triggers`. GitLab's API offers far more than just triggering pipelines, in fact it can perform every action that can be performed manually using the Web-UI. For further reading look up [GitLab's API Documentation](https://docs.gitlab.com/ee/api/README.html).
