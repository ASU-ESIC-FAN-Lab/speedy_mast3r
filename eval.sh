# 7Scenes
# scene in 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs'
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes', subscene='chess', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7scenes/chess/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes', subscene='fire', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7scenes/fire/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes', subscene='heads', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7scenes/heads/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes', subscene='office', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7scenes/office/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes', subscene='pumpkin', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7scenes/pumpkin/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes', subscene='redkitchen', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7scenes/redkitchen/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocSevenScenes('/path/to/prepared/7-scenes', subscene='stairs', pairsfile='APGeM-LM18_top20', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/7scenes/stairs/loc --use_amp --single_loop --use_tensorrt


# InLoc
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocInLoc('/path/to/prepared/InLoc', pairsfile='pairs-query-netvlad40-temporal', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/inloc/top1/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocInLoc('/path/to/prepared/InLoc', pairsfile='pairs-query-netvlad40-temporal', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/inloc/top20/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocInLoc('/path/to/prepared/InLoc', pairsfile='pairs-query-netvlad40-temporal', topk=40)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/inloc/top40/loc --use_amp --single_loop --use_tensorrt


# Aachen-Day-Night
# scene in 'day' 'night'
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/aachenv11', subscene='day', pairsfile='fire_top50', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/aachen/top1/day/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/aachenv11', subscene='day', pairsfile='fire_top50', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/aachen/top20/day/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/aachenv11', subscene='day', pairsfile='fire_top50', topk=40)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/aachen/top40/day/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/aachenv11', subscene='night', pairsfile='fire_top50', topk=1)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/aachen/top1/night/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/aachenv11', subscene='night', pairsfile='fire_top50', topk=20)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/aachen/top20/night/loc --use_amp --single_loop --use_tensorrt
python3 visloc.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "VislocAachenDayNight('/path/to/prepared/aachenv11', subscene='night', pairsfile='fire_top50', topk=40)" --pixel_tol 5 --pnp_mode poselib --reprojection_error_diag_ratio 0.008 --output_dir /path/to/output/aachen/top40/night/loc --use_amp --single_loop --use_tensorrt


# MegaDepth
python relpose.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "RelPoseMegaDepth1500('/path/to/prepared/reloc3r_dataset/megadepth1500', pairsfile='megadepth_test_pairs', resolution=(512,384))" --pose_estimator cv2 --use_amp --single_loop --use_tensorrt


# ScanNet
python relpose.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --dataset "RelPoseScanNet1500('/path/to/prepared/reloc3r_dataset/scannet1500', pairsfile='test', resolution=(512,384))" --pose_estimator cv2 --use_amp --single_loop --use_tensorrt
