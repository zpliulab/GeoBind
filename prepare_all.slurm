#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition=debug
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16000
#SBATCH --time 20:00:00
#SBATCH --array=0-999
#SBATCH --output=exelogs/_GeoBind_precompute.%A_%a.out
#SBATCH --error=exelogs/_GeoBind_precompute.%A_%a.err
i=0

while read p; do
  if [ $(( i % 1000 )) == ${SLURM_ARRAY_TASK_ID} ]; then
    echo $p
    FIELD1=$(echo $p| cut -d" " -f1)
    start_time=$(date +%s)
    echo $FIELD1
    python prepare_one.py --pdbid $FIELD1 --ligand MG
    end_time=$(date +%s)
    cost_time=$[ $end_time-$start_time ]
    #echo "$i	${SLURM_ARRAY_TASK_ID}  $p	$cost_time"
  fi
  i=$((i+1))

#done < ./Dataset_lists/GeoBind/DNA-719_Train.txt
#done < ./Dataset_lists/GeoBind/DNA-179_Test.txt
#done < ./Dataset_lists/GeoBind/RNA-663_Train.txt
#done < ./Dataset_lists/GeoBind/RNA-157_Test.txt

#done < ./Dataset_lists/Ligands_by_DELIA_ATPbind/HEM-175_Train.txt
#done < ./Dataset_lists/Ligands_by_DELIA_ATPbind/HEM-96_Test.txt
#done< ./Dataset_lists/Ligands_by_DELIA_ATPbind/ATP-388_Train.txt
#done< ./Dataset_lists/Ligands_by_DELIA_ATPbind/ATP-41_Test.txt
#done < Dataset_lists/Ligands_by_DELIA_ATPbind/MN-440_Train.txt
#done < Dataset_lists/Ligands_by_DELIA_ATPbind/MN-144_Test.txt
#done < Dataset_lists/Ligands_by_DELIA_ATPbind/MG-1194_Train.txt
done < Dataset_lists/Ligands_by_DELIA_ATPbind/MG-651_Test.txt
#done < Dataset_lists/Ligands_by_DELIA_ATPbind/CA-1022_Train.txt
#done < Dataset_lists/Ligands_by_DELIA_ATPbind/CA-515_Test.txt