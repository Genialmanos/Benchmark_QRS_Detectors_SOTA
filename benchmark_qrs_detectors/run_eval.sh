#!/bin/bash
# Liste des DATASET à tester
#declare -a datasets=( "mit-bih-noise-stress-test-e00" "mit-bih-noise-stress-test-e06" "mit-bih-noise-stress-test-e12" "mit-bih-noise-stress-test-e18" "mit-bih-noise-stress-test-e24" )
#"mit-bih-supraventricular-arrhythmia" "mit-bih-arrhythmia" "european-stt" "mit-bih-long-term-ecg")
#declare -a datasets=("mit-bih-long-term-ecg")
declare -a datasets=("mit-bih-noise-stress-test-e24" "mit-bih-noise-stress-test-e18" "mit-bih-noise-stress-test-e12" "mit-bih-noise-stress-test-e06" "mit-bih-noise-stress-test-e00" "mit-bih-supraventricular-arrhythmia" "mit-bih-arrhythmia" "european-stt" "mit-bih-long-term-ecg")
# "mit-bih-noise-stress-test-e24" "mit-bih-noise-stress-test-e18" "mit-bih-noise-stress-test-e12" "mit-bih-noise-stress-test-e06" "mit-bih-noise-stress-test-e00" "mit-bih-supraventricular-arrhythmia" "mit-bih-arrhythmia" )
# Parcours de la liste des DATASET
for dataset in "${datasets[@]}"
do
  echo "Lancement de l'algo avec le DATASET : $dataset"
  time_result=$( { time make evaluation DATASET="$dataset" ALGO="wavelet_coef" TOLERANCE=100 ; } 2>&1 | grep real | awk '{print $2}')
  echo "Temps d'exécution pour le DATASET $dataset : $time_result secondes"
done