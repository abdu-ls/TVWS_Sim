use Google Colab: https://colab.research.google.com/

or Linux Virtual Machine, if so the prerequisite are as follows:

Step 1: Create an isolated workspace so nothing conflicts: 
1. python3.10 -m venv iomt_env 
2. source iomt_env/bin/activate

Step 2: Upgrade pip (the package installer) first
pip install --upgrade pip

Step 3: Install PyTorch
pip install torch torchvision

Step 4: Install the simulation and analysis tools
pip install simpy numpy pandas matplotlib seaborn

Step 5: Confirm everything installed correctly

python -c "import torch, simpy, numpy, pandas, matplotlib; print('All good')"


********* Now run the file ************
python s_sensing_.py

NOTE: it takes 6 hours+ to complete. all plots will be generated for the results.
