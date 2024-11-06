# aind-vr-foraging-analysis
A repository with analysis code for the aind-vr-foraging experiment

# Save environment settings
pip freeze > requirements.txt

# Install the environment using create environment in Visual Studio and selecting requirements.txt

# Cloning harp-tech devices and Allen institute devices (current version) for parsing the data streams. 
## Create a subfolder somewhere called harp-tech and clone all these devices. 
git clone https://github.com/AllenNeuralDynamics/harp.device.sniff-detector
git clone https://github.com/AllenNeuralDynamics/harp.device.lickety-split
git clone https://github.com/harp-tech/device.analoginput
git clone https://github.com/harp-tech/device.olfactometer
git clone https://github.com/harp-tech/device.behavior

# Use the following command to install in editable mode
pip install -e ./
