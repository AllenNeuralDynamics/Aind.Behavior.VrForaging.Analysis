# aind-vr-foraging-analysis
A repository with analysis code for the aind-vr-foraging experiment

# Save environment settings
pip freeze > requirements.txt

# How to install the environment
## (recommended) Use UV and type these commands in the terminal
* uv venv
* uv sync --extra linters (if you want them)
* cd .venv\Scripts\activate.ps1
* uv pip install -e ./ in terminal

## Alternative:
* Install the environment using create environment in Visual Studio and selecting requirements.txt
* Activate environment
* Run pip install -e ./ in terminal
