# DS-Competition

[![Pylint](https://github.com/thomasthaddeus/DS-Competition/actions/workflows/pylint.yml/badge.svg?branch=dev)](https://github.com/thomasthaddeus/DS-Competition/actions/workflows/pylint.yml)

This repository holds in progress work.

## Distribution of Work

1. **Eli - Data Collection and Preprocessing Specialist**: This person would be responsible for collecting the necessary data for the project, cleaning it, and transforming it into a format that can be used for model training. They would also handle any necessary data augmentation.

2. **Thad - Feature Engineer**: This person would be responsible for creating new features from the existing data that might help improve the model's performance. They would work closely with the Data Collection and Preprocessing Specialist to understand the data and come up with effective features.

3. **Model Developer 1**: This person would be responsible for selecting a suitable model, training it, and tuning its parameters. They would work closely with the Feature Engineer to understand the features and how they can be used in the model.

4. **Model Developer 2**: This person would also be responsible for model development. Having two people on this task allows for parallel experimentation with different models or different sets of parameters, which can speed up the process and potentially lead to better results.

5. **Validation and Testing Specialist**: This person would be responsible for evaluating the model's performance using a validation set and making adjustments to the model if necessary. They would work closely with the Model Developers to understand the models and how they can be improved.

6. **Person 6 - Submission and Documentation Manager / Infrastructure Manager**: This person would be responsible for submitting the team's entries to the competition, documenting the team's work, and managing the infrastructure needed for model training. This includes keeping track of the different models that were tried, the features that were used, and the performance of each model. They would also handle any necessary setup and management of cloud resources, and manage the team's code using a version control system like Git.

## Users

1. Thad
2. Eli
3. Nicholas

## Project Structure

<!-- Description of the project's directory structure and main files. -->

- `.github`: don't touch this folder
- `/data`: all data should be stored here
- `/models`: store learning models here
- `/notebooks`: put all notebooks here under your folder
- `/src`: any source code you need to import for your notebook to work

<code><div>
<h3>Directory Tree</h3><p>
<a href="./">.</a><br>
├── <a href="./data/">data</a><br>
│   ├── <a href="./data/eval_student_summaries/">eval_student_summaries</a><br>
│   │   ├── <a href="./data/eval_student_summaries/prompts_test.csv">prompts_test.csv</a><br>
│   │   ├── <a href="./data/eval_student_summaries/prompts_train.csv">prompts_train.csv</a><br>
│   │   ├── <a href="./data/eval_student_summaries/sample_submission.csv">sample_submission.csv</a><br>
│   │   ├── <a href="./data/eval_student_summaries/summaries_test.csv">summaries_test.csv</a><br>
│   │   └── <a href="./data/eval_student_summaries/summaries_train.csv">summaries_train.csv</a><br>
│   └── <a href="./data/json/">json</a><br>
├── <a href="./LICENSE">LICENSE</a><br>
├── <a href="./models/">models</a><br>
├── <a href="./notebooks/">notebooks</a><br>
│   └── <a href="./notebooks/sample_notebk.ipynb">sample_notebk.ipynb</a><br>
├── <a href="./README.md">README.md</a><br>
├── <a href="./requirements.txt">requirements.txt</a><br>
├── <a href="./sitemap.html">sitemap.html</a><br>
├── <a href="./src/">src</a><br>
│   ├── <a href="./src/evaluation/">evaluation</a><br>
│   ├── <a href="./src/prep/">prep</a><br>
│   │   ├── <a href="./src/prep/data_prep.py">data_prep.py</a><br>
│   │   └── <a href="./src/prep/text_prep.py">text_prep.py</a><br>
│   ├── <a href="./src/scripts/">scripts</a><br>
│   └── <a href="./src/visualize/">visualize</a><br>
└── <a href="./tests/">tests</a><br>
</div></code>

## Setup and Installation

Instructions for setting up and installing any necessary software or libraries.

If you want to use weights and biases here is the [link](https://wandb.ai/site/research)

## Usage

Instructions for how to run the code.

[How to setup the virtual environment.](./docs/venv_setup.md)

## License

[MIT](./LICENSE)
