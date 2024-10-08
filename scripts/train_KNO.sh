# tube
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c geo | tee ./log/KNO/train/KNO_tube_geo.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c bc | tee ./log/KNO/train/KNO_tube_bc.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop | tee ./log/KNO/train/KNO_tube_prop.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c bc_geo | tee ./log/KNO/train/KNO_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_geo | tee ./log/KNO/train/KNO_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_bc | tee ./log/KNO/train/KNO_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_bc_geo | tee ./log/KNO/train/KNO_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c Re_ReD | tee ./log/KNO/train/KNO_tgv_Re_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c Re | tee ./log/KNO/train/KNO_tgv_Re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c ReD | tee ./log/KNO/train/KNO_tgv_ReD.out


# cavity
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c ReD | tee ./log/KNO/train/KNO_cavity_ReD.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c bc | tee ./log/KNO/train/KNO_cavity_bc.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c re | tee ./log/KNO/train/KNO_cavity_re.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c ReD_bc_re | tee ./log/KNO/train/KNO_cavity_ReD_bc_re.out

# NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ibc | tee ./log/KNO/train/KNO_nsch_ibc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ca | tee ./log/KNO/train/KNO_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c mob | tee ./log/KNO/train/KNO_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c phi | tee ./log/KNO/train/KNO_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c re | tee ./log/KNO/train/KNO_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c eps | tee ./log/KNO/train/KNO_nsch_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps | tee ./log/KNO/train/KNO_ibc_phi_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ca_mob_re_eps | tee ./log/KNO/train/KNO_nsch_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c phi_ibc | tee ./log/KNO/train/KNO_nsch_phi_ibc.out


# cylinder
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cylinder_rRE_rBC.yaml -c rRE_rBC | tee ./log/KNO/train/KNO_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cylinder_rRE_rBC.yaml -c rRE | tee ./log/KNO/train/KNO_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cylinder_rRE_rBC.yaml -c rBC | tee ./log/KNO/train/KNO_cylinder_rBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_hills_rRE.yaml -c rRE | tee ./log/KNO/train/KNO_hills_rRE.out