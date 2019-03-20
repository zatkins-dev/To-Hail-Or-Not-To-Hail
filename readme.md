## Setup
Requires python>=3.5

To install requisite packages, run the command
```
$ pip3 install -r modeling/requirements.txt
```

## Usage
To generate a model, a module-scope script is provided. Run the command:
```
$ python3 -m modeling
```
Then, select between a full model (all rows in dataset, _very_ expensive) and a test model (user-set number of rows from dataset).

Due to file hosting restrictions, a key file is required to download new data.

## Data Source:
https://www.kaggle.com/noaa/gsod
