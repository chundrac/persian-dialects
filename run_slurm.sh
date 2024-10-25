module use /sapps/etc/modules/start/
module load generic

for K in {1..8}
do
	for k in {0..4}
	do
		sbatch run.sh python3 enc_dec_crossval.py $K 0 $k
	done
done
