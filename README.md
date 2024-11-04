# TFM Dark Higgs

## Analysis steps

### First, enter the mkShapesRDF environment

git clone https://github.com/giorgiopizz/mkShapesRDF

cd mkShapesRDF

./install.sh

bash

source start.sh

And then proceed with the analysis:

* Compile 

mkShapesRDF -c 1

* Running on local: 

mkShapesRDF -o 0 -f . -b 0 -l10

* Running on batch (CONDOR): 

mkShapesRDF -o 0 -f . -b 1

* Check if there are filled jobs

mkShapesRDF -o 1 -f .

* Resubmit jobs

mkShapesRDF -o 1 -f . -r 1

* Merge all root files. 

mkShapesRDF -o 2 -f .

* For plotting the variables

mkPlot --inputFile rootFiles__darkHiggs2018_v7/mkShapes__darkHiggs2018_v7.root --showIntegralLegend 1

(--showIntegralLegend 1 to show yields)


## Combine environment 

It is important to note that this analysis was done with CMSSW_13_2_10, so other versions could provide different results...

cd ./combine/CMSSW_13_2_10/src/HiggsAnalysis/CombinedLimit/

cmsenv

## To obtain datacards and produce limits

mkDatacards

* Now use python scripts "mk_Limits.py" and "brazil_band.py" (hardcode these scripts to change the mx, mZ and ms domains). After that, running within a combine framework, run:

python3 mk_Limits.py

## To train Machine Learning models

Enter the Machine_Learning folder, where the doTrainDF.py file can be found (enter this file to understand its functionalities)

python3 doTrainDF.py

* Within the Models folder, you will find all models used for my TFM

## Comments on the different files

* Cuts: We have four files, "cuts.py" with selection criteria for the WW analysis, "cuts-CR-higgs.py" with the control regions for the dark-Higgs study, "cuts-signal.py" for the dark-Higgs signal, and "cuts-all_regions_higgs.py" which containts both.

* Nuisances: "nuisances.py" without systematic uncertainties, "nuisances_ALL.py" with all nuisances from a previous analysis and "nuisances_all.py" being a revised version

* Plot, Samples and Aliases: two files each, allowing to introduce the dark-Higgs signal or not

* Variables: three files, "variables.py" containing the essential variables for the WW analysis, "variables_DH.py" with the newly added variables and few machine learning discriminants, and "variables_DH_discriminants.py" with all discriminants for the parametric DNN and RF.

