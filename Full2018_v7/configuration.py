"""
Configuration file for mkShapesRDF script.

It's the only necessary python configuration file, all the other files are imported and defined by this one.

"""

treeName = 'Events'

#: tag used to identify the configuration folder version
#tag = 'darkHiggs2018_v7-DH-discrimininants_RFandDNN_mx100_ms160_all'
#tag = 'darkHiggs2018_v7-DH-discrimininants_dnn_high'
tag = 'darkHiggs2018_v7-All_regions-diff_charge'

#: file to use as runner script, default uses mkShapesRDF.shapeAnalysis.runner, otherwise specify path to script
runnerFile = "default"

#: output file name
outputFile = "mkShapes__{}.root".format(tag)

#: path to ouput folder
outputFolder  = "../../../../../../../../eos/user/v/victorr/rootFiles/rootFiles__{}".format(tag)


#: path to batch folder (used for condor submission)
#batchFolder = "condor_DH-discrimininants_dnn_high"
batchFolder = "condor_All_regions-diff_charge"


#: path to configuration folder (will contain all the compiled configuration files)
configsFolder = "configsCR"


# file with TTree aliases
aliasesFile = 'aliases.py'

# file with list of variables
variablesFile = 'variables.py'
#variablesFile = 'variables_DH.py'
#variablesFile = 'variables_DH_discriminants.py'

# file with list of cuts
# for CRs
cutsFile = 'cuts.py'
# for higgs signal CRs
#cutsFile = 'cuts-CR-higgs.py'
# for higgs signals
#cutsFile = 'cuts-signal.py' 
# all higgs regions
#cutsFile = 'cuts-all_regions_higgs.py'

# file with list of samples
# for CRs
samplesFile = 'samples.py' 
# for higgs signals
#samplesFile = 'samples-signal.py' 

# file with list of samples
# for CRs
plotFile = 'plot.py'
# for higgs signals
#plotFile = 'plot-signal.py' 

# luminosity to normalize to (in 1/fb)
lumi = 59.8


# used by mkDatacards to define output directory for datacards
outputDirDatacard = 'datacards_2016'

# structure file for datacard
structureFile = 'structure.py'

# nuisances file for mkDatacards and for mkShape
nuisancesFile = 'nuisances_all.py'
#nuisancesFile = 'nuisances_ALL.py'


#: path to folder where to save plots
#plotPath = "/eos/user/v/victorr/www/Main_files/DH-discrimininants_dnn_high"
plotPath = "/eos/user/v/victorr/www/Main_files/All_regions-diff_charge"


#: list of imports to import when compiling the whole configuration folder, it should not contain imports used by configuration.py
imports = ["os", "glob", ("collections", "OrderedDict"), "ROOT"]


#: list of files to compile
filesToExec = [
    samplesFile,
    aliasesFile,
    variablesFile,
    cutsFile,
    plotFile,
    nuisancesFile,
    structureFile,
]

#: list of variables to keep in the compiled configuration folder
varsToKeep = [
    "batchVars",
    "outputFolder",
    "batchFolder",
    "configsFolder",
    "outputFile",
    "runnerFile",
    "tag",
    "samples",
    "aliases",
    "variables",
    ("cuts", {"cuts": "cuts", "preselections": "preselections"}),
    ("plot", {"plot": "plot", "groupPlot": "groupPlot", "legend": "legend"}),
    "nuisances",
    "structure",
    "lumi",
]

#: list of variables to keep in the batch submission script (script.py)
batchVars = varsToKeep[varsToKeep.index("samples") :]


varsToKeep += ['plotPath']
