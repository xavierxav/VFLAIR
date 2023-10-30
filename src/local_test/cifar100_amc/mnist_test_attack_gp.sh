
echo 'GradPerturb bu amc Begin'
# #SBATCH --qos high
# DP 0.01
python cifar100_amc.py --configs cifar100_amc/gp

# DP 0.1
sed -i 's/"perturb_epsilon": 0.01/"perturb_epsilon": 0.1/g' ./configs/cifar100_amc/gp.json
python cifar100_amc.py --configs cifar100_amc/gp

# DP 1.0
sed -i 's/"perturb_epsilon": 0.1/"perturb_epsilon": 1.0/g' ./configs/cifar100_amc/gp.json
python cifar100_amc.py --configs cifar100_amc/gp

# DP 10.0
sed -i 's/"perturb_epsilon": 1.0/"perturb_epsilon": 10.0/g' ./configs/cifar100_amc/gp.json
python cifar100_amc.py --configs cifar100_amc/gp

# DP 50.0
sed -i 's/"perturb_epsilon": 10.0/"perturb_epsilon": 50.0/g' ./configs/cifar100_amc/gp.json
# python cifar100_amc.py --configs cifar100_amc/gp

# DP 10.0
sed -i 's/"perturb_epsilon": 50.0/"perturb_epsilon": 100.0/g' ./configs/cifar100_amc/gp.json
# python cifar100_amc.py --configs cifar100_amc/gp

sed -i 's/"perturb_epsilon": 100.0/"perturb_epsilon": 0.01/g' ./configs/cifar100_amc/gp.json

echo 'GradPerturb End'

