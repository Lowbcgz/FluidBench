# tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/KNO/test/KNO_tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/KNO/test/KNO_tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/KNO/test/KNO_tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/KNO/test/KNO_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/KNO/test/KNO_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/KNO/test/KNO_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/KNO/test/KNO_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c Re_ReD --test | tee ./log/KNO/test/KNO_tgv_Re_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c Re --test | tee ./log/KNO/test/KNO_tgv_Re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c ReD --test | tee ./log/KNO/test/KNO_tgv_ReD.out

# cavity
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/KNO/test/KNO_cavity_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/KNO/test/KNO_cavity_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/KNO/test/KNO_cavity_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/KNO/test/KNO_cavity_ReD_bc_re.out

# NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ibc --test | tee ./log/KNO/test/KNO_nsch_ibc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ca --test | tee ./log/KNO/test/KNO_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c mob --test | tee ./log/KNO/test/KNO_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c phi --test | tee ./log/KNO/test/KNO_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c re --test | tee ./log/KNO/test/KNO_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c eps --test | tee ./log/KNO/test/KNO_nsch_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps --test | tee ./log/KNO/test/KNO_ibc_phi_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ca_mob_re_eps --test | tee ./log/KNO/test/KNO_nsch_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c phi_ibc --test | tee ./log/KNO/test/KNO_nsch_phi_ibc.out

# cylinder
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cylinder_rRE_rBC.yaml -c rRE_rBC --test | tee ./log/KNO/test/KNO_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cylinder_rRE_rBC.yaml -c rRE --test | tee ./log/KNO/test/KNO_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/KNO/config_cylinder_rRE_rBC.yaml -c rBC --test | tee ./log/KNO/test/KNO_cylinder_rBC.out