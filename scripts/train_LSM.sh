# tube
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c geo | tee ./log/train/LSM_tube_geo.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c bc | tee ./log/train/LSM_tube_bc.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop | tee ./log/train/LSM_tube_prop.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c bc_geo | tee ./log/train/LSM_tube_bc_geo.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop_geo | tee ./log/train/LSM_tube_prop_geo.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop_bc | tee ./log/train/LSM_tube_prop_bc.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop_bc_geo | tee ./log/train/LSM_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_TGV_all.yaml -c all | tee ./log/train/LSM_tgv_all.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_TGV_all.yaml -c single | tee ./log/train/LSM_tgv_single.out

# # cavity
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c ReD | tee ./log/train/LSM_cavity_ReD.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c bc | tee ./log/train/LSM_cavity_bc.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c re | tee ./log/train/LSM_cavity_re.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c ReD_bc_re | tee ./log/train/LSM_cavity_ReD_bc_re.out

# NSCH
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c bc | tee ./log/train/LSM_nsch_bc.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c ca | tee ./log/train/LSM_nsch_ca.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c mob | tee ./log/train/LSM_nsch_mob.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c phi | tee ./log/train/LSM_nsch_phi.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c pre | tee ./log/train/LSM_nsch_pre.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c re | tee ./log/train/LSM_nsch_re.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c uv0 | tee ./log/train/LSM_nsch_uv0.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c bc_ca_mob_phi_pre_re_uv0 | tee ./log/train/LSM_nsch_bc_ca_mob_phi_pre_re_uv0.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c ca_mob_re | tee ./log/train/LSM_nsch_ca_mob_re.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_NSCH_ALL.yaml -c phi_pre_uv0 | tee ./log/train/LSM_nsch_phi_pre_uv0.out

# Darcy
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_Darcy_ar_as.yaml -c ar | tee ./log/train/LSM_darcy_ar.out
# CUDA_VISIBLE_DEVICES=1 python train.py ./config/LSM/config_Darcy_ar_as.yaml -c as | tee ./log/train/LSM_darcy_as.out