import os
from brazil_band import getLimits, plotUpperLimits

# Directories names
datacards_dir = f"datacards_files"
root_dir = f"root_files"
limit_dir = f"limit_files"

# Mass values
mhs_values = ['200']
mDM_values = ['300']
#mZp_values = ['200','300','400','500','800','1000','1200','1500','2000','2500']
mZp_values = ['800','1000','1200','1500','2000','2500']

# Function to execute the shell commands
def run_commands(hs, DM, Zp):
    # Create directories for saving files
    os.makedirs(datacards_dir, exist_ok=True)
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(limit_dir, exist_ok=True)

    # Create filenames incorporating hs, DM, and Zp
    comb_card_filename = os.path.join(datacards_dir, f"DH_CombCard_mhs_{hs}_mx_{DM}_mz_{Zp}.txt")
    root_filename = os.path.join(root_dir, f"DH_mhs_{hs}_mx_{DM}_mz_{Zp}.root")
    limit_filename = os.path.join(limit_dir, f"Limit_mhs_{hs}_mx_{DM}_mz_{Zp}.txt")

    combine_cmd = f"combineCards.py higgs_sr_/evaluate_normal_dnn/datacard_DH_mhs_{hs}_mx_{DM}_mZp_{Zp}.txt " \
                  f"DY_cr_/events/datacard_DH_mhs_{hs}_mx_{DM}_mZp_{Zp}.txt " \
                  f"tt_cr_/events/datacard_DH_mhs_{hs}_mx_{DM}_mZp_{Zp}.txt " \
                  f"WW_cr_/events/datacard_DH_mhs_{hs}_mx_{DM}_mZp_{Zp}.txt > {comb_card_filename}"

    text2workspace_cmd = f"text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose " \
            f"--PO 'map=.*/DH_mhs_{hs}_mx_{DM}_mZp_{Zp}:r[1,0,10]' {comb_card_filename} -o {root_filename}"

    combine_limit_cmd = f"combine -M AsymptoticLimits -t -1 --expectSignal 1 --cminDefaultMinimizerStrategy 0 {root_filename} &> {limit_filename}"

    os.system(combine_cmd)
    os.system(text2workspace_cmd)
    os.system(combine_limit_cmd)

# Iterate through the mass values and run commands
for hs in mhs_values:
    for DM in mDM_values:
        for Zp in mZp_values:
            # Skipping specific combinations
            if (DM == '300' and Zp in ['200', '300', '400', '500']) or \
               (hs == '300' and DM == '300' and Zp in ['1500', '2000', '2500']) or \
               (hs == '300' and DM == '200' and Zp in ['200', '300', '400', '2000', '2500']) or \
               (hs == '300' and DM == '150' and Zp in ['200', '300', '2000', '2500']) or \
               (hs == '300' and DM == '100'):
                continue
            run_commands(hs, DM, Zp)

# MAIN
def main():
    labels = []
    values = []
    for hs in mhs_values:
        for DM in mDM_values:
            for Zp in mZp_values:
                # Skipping specific combinations
                if (DM == '300' and Zp in ['200', '300', '400', '500']) or \
                   (hs == '300' and DM == '300' and Zp in ['1500', '2000', '2500']) or \
                   (hs == '300' and DM == '200' and Zp in ['200', '300', '400', '2000', '2500']) or \
                   (hs == '300' and DM == '150' and Zp in ['200', '300', '2000', '2500']) or \
                   (hs == '300' and DM == '100'):
                    continue
                label = os.path.join(limit_dir, f"Limit_mhs_{hs}_mx_{DM}_mz_{Zp}.txt") 
                labels.append(label)
                values.append(int(Zp))

    if labels:
        plotUpperLimits(labels, values, mhs_values[0], mDM_values[0])

if __name__ == '__main__':
    main()
