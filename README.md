# ModelCI
Git theta based continuous integration of ML models

#### Requirements
1. Git-Theta - [Instructions to install](https://github.com/r-three/git-theta?tab=readme-ov-file#installing-git-theta)
2. Deepchecks - [Instructions to install](https://docs.deepchecks.com/stable/getting-started/installation.html#installation)
3. Pytorch - [Instructions to install](https://pytorch.org/get-started/locally/)

#### Usage
ModelCI can be used for automated vetting and integration of machine learning models. Currently, only image classification
models are supported.

##### Setup
1. Clone the git repository
    ```
    git clone https://github.com/brijml/ModelCI.git
    ```
2. Create a new local branch
    ```
    git checkout -b <branch-name>
    ```

##### Local Testing
1. Train an image classification model on a dataset. 
2. Use `deepchecks-suite.py` to execute the test suite locally on the model that includes,
   * Class Performance Check - Measure the Precision and Recall score for each class and pass the check if the values cross 0.2 for every class.
   * Simple Model Performance - Measure the F1 score for every class for the model and a simple model which predicts the class randomly and pass the check only if the gain i.e. the ratio (model score - simple score/ perfect score - simple score) is above 0.99 for every class.
   * Weak Segments Performance Check - Measure model accuracy for different range of image property values like the mean brightness.

##### Model Integration

We make use of Github Actions to execute the same test suite in an ubuntu container upon every push of the code.

1. Save the model and use Git-theta for tracking the model ([Instructions Here!](https://github.com/r-three/git-theta?tab=readme-ov-file#tracking-a-mode)]) 
2. Add and Commit the model
   ```
   git add model.pt
   git commit -m "<comnit-message>"
   ```
3. Merge the branch with the main branch
    ```
    git checkout main
    git merge <branch-name>
    ```
4. Push the model to the github repo
    ```
    git push origin main
    ```
5. Upon model push, Github Action Workflow will execute the `deepcheck-suite.py` and upload the artifacts to Github
6. The running workflow can be [viewed here](https://github.com/brijml/ModelCI/actions/workflows/deepcheck-workflow.yml)
7. Click on the `build` job and locate the `Archive the Deepcheck Results` step to download the artifacts to your desktop to view the results

#### Demo Video

[![Follow Mohamed El-Qassas GitHub](https://drive.google.com/thumbnail?id=1OgVkxgI0xvXhELqFIAzi44mNQMoKuVYb)](https://drive.google.com/file/d/1OgVkxgI0xvXhELqFIAzi44mNQMoKuVYb/preview)


