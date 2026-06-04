# 1. Project Objective

## Marketing spend is across various channels, both online and offline. It is hard to measure the attribution per channel and allocate the budget to reach to max ROI. 

## Therefore, establish a Bayesian MMM to figure out the below questions:
## - the current ROI and marginal ROI per channel
## - Identify which marketing channels are currently saturated (diminishing returns).
## - Identify which channels have room to grow (under-saturated).
## - Provide actionable, tactical recommendations for next week's budget allocation (e.g., "Decrease Paid Search spend because it hit saturation; Increase Display spend by X% to capture remaining potential").

### Data source: data is generated from generate_synthetic.py and store in Supabase

# 2. Environment Set up

## 1) Python version and virtual environment set up

Run below to check the default Python version:

```bash
python --version
```

Initialise the Poetry Project

```bash
poetry init
```
You will populate the package name, Version, Description, etc.

After that, it will pop up below questions and select 'no':

Would you like to define your main dependencies interactively?
Would you like to define your development dependencies interactively?

Run below to set up Python environment and activate
```bash
poetry env use python
```
In pyprojct.toml, set the below version to ensure using Python 3.12:
```bash
requires-python = ">=3.12, <3.15"
```

```bash
poetry env activate
```
 
Using poetry to add libraries:
```bash
poetry add numpy pandas matplotlib seaborn datetime pymc arvia supabse streamlit
```
In pyproject.toml, set up package-mode
```
[tool.poetry]
package-mode = false
```

```bash
poetry install
```
## 2) Interpreter set up
Go to the IDE bottom right - Interpreter Setting - Add Interpreter - Local Interpreter
Select **Poetry**

(how to add picture?)

You will see the installed packages under that interpreter.

## 3) Install Visual Code C++ to speed up PyMC processing

Install Visual Code then add the below path to Environment Variable - System Variable
```
C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
```

## 3. Git Hygiene (.gitignore)
Recommend adding below files to .gitignore to avoid uploading unnecessary or sensitivity data
```bash
.venv/
.idea/
**/__pycache__/
**.pyc
data/synthetic/*.csv
**/.ipynb_checkpoints/
```


what can we do after that

methodolodgy & Feature engineer

performance & insights

key findings

future improvements