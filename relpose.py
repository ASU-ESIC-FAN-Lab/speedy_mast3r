import argparse
import os
from collections import defaultdict
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import opencv_to_colmap_intrinsics
from dust3r.datasets import get_data_loader

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r_relpose.datasets import *

from tqdm import tqdm
import poselib, cv2, pycolmap
import tensorrt as trt


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="visloc dataset to eval")
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])

    parser.add_argument("--confidence_threshold", type=float, default=1.001,
                        help="confidence values higher than threshold are invalid")
    parser.add_argument('--pixel_tol', default=5, type=int)

    parser.add_argument("--coarse_to_fine", action='store_true', default=False,
                        help="do the matching from coarse to fine")
    parser.add_argument("--max_image_size", type=int, default=None,
                        help="max image size for the fine resolution")
    parser.add_argument("--c2f_crop_with_homography", action='store_true', default=False,
                        help="when using coarse to fine, crop with homographies to keep cx, cy centered")

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--pose_estimator", type=str, default="cv2", choices=['cv2', 'poselib', 'pycolmap'],
                        help="Pose lib to use")
    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument("--reprojection_error", type=float, default=5.0, help="pnp reprojection error")
    parser_reproj.add_argument("--reprojection_error_diag_ratio", type=float, default=None,
                               help="pnp reprojection error as a ratio of the diagonal of the image")

    parser.add_argument("--max_batch_size", type=int, default=48,
                        help="max batch size for inference on crops when using coarse to fine")
    parser.add_argument("--pnp_max_points", type=int, default=100_000, help="pnp maximum number of points kept")
    parser.add_argument("--viz_matches", type=int, default=0, help="debug matches")

    parser.add_argument("--output_dir", type=str, default=None, help="output path")
    parser.add_argument("--output_label", type=str, default='', help="prefix for results files")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument("--use_amp", action='store_true', default=False, help="Use amp in fastnn")
    parser.add_argument("--single_loop", action='store_true', default=False, help="Use single-loop matmul")
    parser.add_argument("--use_tensorrt", action='store_true', default=False, help="Apply tensorrt to the model")
    return parser


class TensorWrapper:
    _data = None

    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        else:
            return NotImplemented


class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 3, 3)."""
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 3)."""
        return self._data[..., -3:]

    def __repr__(self):
        return f"Pose: {self.shape} {self.dtype} {self.device}"


class Camera(TensorWrapper):
    eps = 1e-4

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6, 8, 10}
        super().__init__(data)

    @classmethod
    def from_calibration_matrix(cls, K: torch.Tensor):
        cx, cy = K[..., 0, 2], K[..., 1, 2]
        fx, fy = K[..., 0, 0], K[..., 1, 1]
        data = torch.stack([2 * cx, 2 * cy, fx, fy, cx, cy], -1)
        return cls(data)

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., 4:6]

    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / self.f.unsqueeze(-2)

    def image2cam(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert 2D pixel corrdinates to 3D points with z=1"""
        assert self._data.shape
        p2d = self.normalize(p2d)
        # iterative undistortion
        return to_homogeneous(p2d)

    def to_cameradict(self, camera_model=None):
        data = self._data.clone()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert data.dim() == 2
        b, d = data.shape
        if camera_model is None:
            camera_model = {6: "PINHOLE", 8: "RADIAL", 10: "OPENCV"}[d]
        cameras = []
        for i in range(b):
            if camera_model.startswith("SIMPLE_"):
                params = [x.item() for x in data[i, 3 : min(d, 7)]]
            else:
                params = [x.item() for x in data[i, 2:]]
            cameras.append(
                {
                    "model": camera_model,
                    "width": int(data[i, 0].item()),
                    "height": int(data[i, 1].item()),
                    "params": params,
                }
            )
        return cameras if self._data.dim() == 2 else cameras[0]

    def __repr__(self):
        return f"Camera {self.shape} {self.dtype} {self.device}"

@torch.no_grad()
def inference(batch, model, device, use_amp=False, events=None): 
    # to device. 
    for view in batch:
        for name in 'img camera_intrinsics camera_pose'.split():  
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    # forward. 
    view1, view2 = batch
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        return model(view1, view2, events=events)

def angle_error_vec(v1, v2, eps=1e-10):
    n = torch.clip(v1.norm(dim=-1) * v2.norm(dim=-1), min=eps)
    v1v2 = (v1 * v2).sum(dim=-1)  # dot product in the last dimension
    return torch.rad2deg(torch.arccos(torch.clip(v1v2 / n, -1.0, 1.0)))

def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.einsum("...ij, ...jk -> ...ik", R1.T, R2)) - 1) / 2
    cos = torch.clip(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.arccos(cos)))

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0, eps=1e-10):
    if isinstance(T_0to1, torch.Tensor):
        R_gt, t_gt = T_0to1[:3, :3], T_0to1[:3, 3]
    else:
        R_gt, t_gt = T_0to1.R, T_0to1.t
    R_gt, t_gt = torch.squeeze(R_gt), torch.squeeze(t_gt)

    # angle error between 2 vectors
    t_err = angle_error_vec(t, t_gt, eps)
    t_err = torch.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if t_gt.norm() < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    r_err = angle_error_mat(R, R_gt)

    return t_err, r_err


def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores

def eval_relative_pose_robust(data, pred, estimator='cv2', **kw):
    T_gt = data["T_0to1"].squeeze(0)
    kp0, kp1 = pred["keypoints0"].squeeze(0), pred["keypoints1"].squeeze(0)
    m0, scores0 = pred["matches0"].squeeze(0), pred["matching_scores0"].squeeze(0)
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}
    
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "camera0": Camera.from_calibration_matrix(data["view0"]["camera_intrinsics"].squeeze(0)).float(),
        "camera1": Camera.from_calibration_matrix(data["view1"]["camera_intrinsics"].squeeze(0)).float(),
        "img_size0": list(data["view0"]['img'].shape[2:])[::-1],
        "img_size1": list(data["view1"]['img'].shape[2:])[::-1],
    }

    assert estimator in ['cv2', 'poselib', 'pycolmap'], f"Invalid estimator: {estimator}"
    if estimator == 'cv2':
        est = OpenCVRelativePoseEstimator(data_, **kw)
    elif estimator == 'poselib':
        est = PoseLibRelativePoseEstimator(data_, **kw)
    elif estimator == 'pycolmap':
        # est = PycolmapTwoViewEstimator(data_, **kw)
        raise Exception("Currently unsupported.")

    if not est["success"]:
        results["rel_pose_error"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M = est["M_0to1"]
        inl = est["inliers"].cpu().detach().numpy()
        t_error, r_error = relative_pose_error(T_gt, M.R, M.t)
        results["rel_pose_error"] = max(r_error, t_error).cpu().detach()
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return results

def from_homogeneous(points, eps=0.0):
    return points[..., :-1] / (points[..., -1:] + eps)

def to_homogeneous(points):
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def OpenCVRelativePoseEstimator(data, ransac_th=0.5, confidence=0.99999):
    solver = cv2.RANSAC
    kpts0, kpts1 = data["m_kpts0"], data["m_kpts1"]
    camera0 = data["camera0"]
    camera1 = data["camera1"]
    M, inl = None, torch.zeros_like(kpts0[:, 0]).bool()

    if len(kpts0) >= 5:
        f_mean = torch.cat([camera0.f, camera1.f]).mean().item()
        norm_thresh = ransac_th / f_mean

        pts0 = from_homogeneous(camera0.image2cam(kpts0)).cpu().detach().numpy()
        pts1 = from_homogeneous(camera1.image2cam(kpts1)).cpu().detach().numpy()

        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            np.eye(3),
            threshold=norm_thresh,
            prob=confidence,
            method=solver,
        )

        if E is not None:
            best_num_inliers = 0
            for _E in np.split(E, len(E) / 3):
                n, R, t, _ = cv2.recoverPose(
                    _E, pts0, pts1, np.eye(3), 1e9, mask=mask
                )
                if n > best_num_inliers:
                    best_num_inliers = n
                    inl = torch.tensor(mask.ravel() > 0)
                    M = Pose.from_Rt(
                        torch.tensor(R).to(kpts0), torch.tensor(t[:, 0]).to(kpts0)
                    )

    estimation = {
        "success": M is not None,
        "M_0to1": M if M is not None else Pose.from_4x4mat(torch.eye(4).to(kpts0)),
        "inliers": inl.to(device=kpts0.device),
    }

    return estimation
        
def PoseLibRelativePoseEstimator(data, ransac_th=2.0):
    pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
    camera0 = data["camera0"]
    camera1 = data["camera1"]
    img_size0 = data["img_size0"]
    img_size1 = data["img_size1"]
    
    M, info = poselib.estimate_relative_pose(
        pts0.cpu().detach().numpy(),
        pts1.cpu().detach().numpy(),
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {
            "max_epipolar_error": ransac_th,
        },
    )
    success = M is not None
    if success:
        M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t)).to(pts0)
    else:
        M = Pose.from_4x4mat(torch.eye(4)).to(pts0)

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": torch.tensor(info.pop("inliers")).to(pts0),
        **info,
    }

    return estimation


def PycolmapTwoViewEstimator(data, ransac_th=4.0):
    options = {**pycolmap.TwoViewGeometryOptions().todict()}
    pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
    camera0 = data["camera0"]
    camera1 = data["camera1"]

    info = pycolmap.two_view_geometry_estimation(
        pts0.cpu().detach().numpy(),
        pts1.cpu().detach().numpy(),
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        options,
    )
    success = info["success"]
    if success:
        R = pycolmap.qvec_to_rotmat(info["qvec"])
        t = info["tvec"]
        M = Pose.from_Rt(torch.tensor(R), torch.tensor(t)).to(pts0)
        inl = torch.tensor(info.pop("inliers")).to(pts0)
    else:
        M = Pose.from_4x4mat(torch.eye(4)).to(pts0)
        inl = torch.zeros_like(pts0[:, 0]).bool()

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": inl,
        "type": str(
            info.get("configuration_type", pycolmap.TwoViewGeometry.UNDEFINED)
        ),
    }

    return estimation

class AUCMetric:
    def __init__(self, thresholds, elements=None):
        self._elements = elements
        self.thresholds = thresholds
        if not isinstance(thresholds, list):
            self.thresholds = [thresholds]

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return cal_error_auc(self._elements, self.thresholds)

def cal_error_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return aucs

def eval_poses(pose_results, auc_ths, key, unit="°"):
    pose_aucs = {}
    best_th = -1
    for th, results_i in pose_results.items():
        pose_aucs[th] = AUCMetric(auc_ths, results_i[key]).compute()
    mAAs = {k: np.mean(v) for k, v in pose_aucs.items()}
    best_th = max(mAAs, key=mAAs.get)

    if len(pose_aucs) > -1:
        print("Tested ransac setup with following results:")
        print("AUC", pose_aucs)
        print("mAA", mAAs)
        print("best threshold =", best_th)

    summaries = {}

    for i, ath in enumerate(auc_ths):
        summaries[f"{key}@{ath}{unit}"] = pose_aucs[best_th][i]
    summaries[f"{key}_mAA"] = mAAs[best_th]

    for k, v in pose_results[best_th].items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        summaries[f"m{k}"] = round(np.median(arr), 3)
    return summaries, best_th

def correct_intrinsic(b_view):
    for i in range(len(b_view['true_shape'])):
        height, width = b_view['true_shape'][i]
        if width < height:
            b_view['camera_intrinsics'][i] = b_view['camera_intrinsics'][i][[1, 0, 2]]

            K = b_view['camera_intrinsics'][i]
            temp = K[0,0]
            K[0,0] = K[1,1]
            K[1,1] = temp

            temp = K[0,2]
            K[0,2] = K[1,2]
            K[1,2] = temp
            b_view['camera_intrinsics'][i] = K

class InferenceWrapper(torch.nn.Module):
    def __init__(self, engine, device):
        super().__init__()
        self.engine = engine
        self.context = engine.create_execution_context()
        self.device = device

    def forward(self, *args):
        for i in range(len(args)):
            self.context.set_input_shape(f"input{i+1}", args[i].shape)
        output = torch.empty(*self.context.get_tensor_shape("output"), device=self.device, dtype=args[0].dtype)

        bindings = [int(input.data_ptr()) for input in args] + [int(output.data_ptr())]
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), bindings[i])
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)

        return output


def optimize_model(model, device):
    gpu_type = torch.cuda.get_device_name(0).replace(' ', '_')
    compiled_dir = f'optimized_modules_{gpu_type}'
    os.makedirs(compiled_dir, exist_ok=True)

    inputs = {'dpt.scratch.refinenet4': (torch.rand(1, 256, 16, 12).cuda(),),
            'dpt.scratch.refinenet3': (torch.rand(1, 256, 32, 24).cuda(), torch.rand(1, 256, 32, 24).cuda()),
            'dpt.scratch.refinenet2': (torch.rand(1, 256, 64, 48).cuda(), torch.rand(1, 256, 64, 48).cuda()),
            'dpt.scratch.refinenet1': (torch.rand(1, 256, 128, 96).cuda(), torch.rand(1, 256, 128, 96).cuda()),
            'dpt.head': (torch.rand(1, 256, 256, 192).cuda(),),
            'head_local_features': (torch.rand(1, 768, 1792).cuda(),)
    }
    dynamic_axes = {'dpt.scratch.refinenet4': {
                    'input1' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'output' : {0 : 'batch_size', 2: 'width', 3: 'height'}},
            'dpt.scratch.refinenet3': {
                    'input1' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'input2' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'output' : {0 : 'batch_size', 2: 'width', 3: 'height'}},
            'dpt.scratch.refinenet2': {
                    'input1' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'input2' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'output' : {0 : 'batch_size', 2: 'width', 3: 'height'}},
            'dpt.scratch.refinenet1': {
                    'input1' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'input2' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'output' : {0 : 'batch_size', 2: 'width', 3: 'height'}},
            'dpt.head': {
                    'input1' : {0 : 'batch_size', 2: 'width', 3: 'height'},
                    'output' : {0 : 'batch_size', 2: 'width', 3: 'height'}},
            'head_local_features': {
                    'input1' : {0 : 'batch_size', 1: 'depth'},
                    'output' : {0 : 'batch_size', 2: 'width'}},
    }
    input_shapes = {'dpt.scratch.refinenet4': {
                    'input1': {'min': (1, 256, 1, 1), 'opt': (1, 256, 16, 12), 'max': (1, 256, 16, 16)}},
            'dpt.scratch.refinenet3': {
                    'input1': {'min': (1, 256, 1, 1), 'opt': (1, 256, 32, 24), 'max': (1, 256, 32, 32)},
                    'input2': {'min': (1, 256, 1, 1), 'opt': (1, 256, 32, 24), 'max': (1, 256, 32, 32)}},
            'dpt.scratch.refinenet2': {
                    'input1': {'min': (1, 256, 1, 1), 'opt': (1, 256, 64, 48), 'max': (1, 256, 64, 64)},
                    'input2': {'min': (1, 256, 1, 1), 'opt': (1, 256, 64, 48), 'max': (1, 256, 64, 64)}},
            'dpt.scratch.refinenet1': {
                    'input1': {'min': (1, 256, 1, 1), 'opt': (1, 256, 128, 96), 'max': (1, 256, 128, 128)},
                    'input2': {'min': (1, 256, 1, 1), 'opt': (1, 256, 128, 96), 'max': (1, 256, 128, 128)}},
            'dpt.head': {
                    'input1': {'min': (1, 256, 1, 1), 'opt': (1, 256, 256, 192), 'max': (1, 256, 256, 256)}},
            'head_local_features': {
                    'input1': {'min': (1, 576, 1792), 'opt': (1, 768, 1792), 'max': (1, 768, 1792)}},
    }
    
    for key, input in inputs.items():
        das = dynamic_axes[key]
        onnx_file_path = f'{compiled_dir}/model.downstream_head1.{key}.onnx'
        engine_file_path = f'{compiled_dir}/model.downstream_head2.{key}.trt'

        # Save engine
        pm = model.downstream_head1
        for sub in key.split('.')[:-1]:
            pm = getattr(pm, sub)
        mod = key.split('.')[-1]
        if not os.path.isfile(engine_file_path):
            torch.onnx.export(getattr(pm, mod),
                            input,
                            onnx_file_path,
                            export_params=True,
                            opset_version=14,
                            do_constant_folding=True,
                            input_names=[f'input{i}' for i in range(1, len(input)+1)],
                            output_names=["output"],
                            dynamic_axes=das,
                            )

            logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            with open(onnx_file_path, 'rb') as f:
                if not parser.parse(f.read()):
                    print("ERROR: Failed to parse ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)   # 1GB workspace size
            config.builder_optimization_level = 5
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 for faster inference (if supported)

            # Update
            profile = builder.create_optimization_profile()
            for i in range(len(input)):
                input_tensor = network.get_input(i)
                profile.set_shape(input_tensor.name, **input_shapes[key][input_tensor.name])
            config.add_optimization_profile(profile)

            engine_bytes = builder.build_serialized_network(network, config)
            with open(engine_file_path, "wb") as f:
                        f.write(engine_bytes)

        # load and run the engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        setattr(pm, mod, InferenceWrapper(engine, device))


    return model


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    conf_thr = args.confidence_threshold
    device = args.device
    pose_estimator = args.pose_estimator
    assert args.pixel_tol > 0
    reprojection_error = args.reprojection_error
    reprojection_error_diag_ratio = args.reprojection_error_diag_ratio
    pnp_max_points = args.pnp_max_points
    viz_matches = args.viz_matches

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    if args.use_tensorrt:
        optimize_model(model, device)
    fast_nn_params = dict(device=device, dist='dot', block_size=2**13, use_amp=args.use_amp, single_loop=args.single_loop)
    dataset = eval(args.dataset)
    data_loader_test = get_data_loader(dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_mem=True,
                             shuffle=False,
                             drop_last=False)
    print('Dataset length: ', len(data_loader_test))

    # start evaluation
    rerrs, terrs = [], []
    print(f'Testing {args.dataset}')

    results = defaultdict(list)
    test_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    pose_results = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for batch in tqdm(data_loader_test):

            events = {
                'start_encode': [],
                'end_encode': [],
                'start_decode': [],
                'end_decode': [],
                'start_downstream_head': [],
                'end_downstream_head': [],
                'start_fastnn': [],
                'end_fastnn': [],
                'start_bnn': [],
                'end_bnn': [],
            }

            res1, res2 = inference(batch, model, device, use_amp=False, events=events)
            desc1, desc2 = (
                res1['desc'].squeeze(0).detach(),
                res2['desc'].squeeze(0).detach(),
            )
            
            start_fastnn = torch.cuda.Event(enable_timing=True)
            end_fastnn = torch.cuda.Event(enable_timing=True)
            start_fastnn.record()
            kp0, kp1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, events=events, **fast_nn_params)
            end_fastnn.record()
            events['start_fastnn'].append(start_fastnn)
            events['end_fastnn'].append(end_fastnn)

            kp0 = torch.as_tensor(np.ascontiguousarray(kp0), dtype=torch.float, device=device).unsqueeze(0)
            kp1 = torch.as_tensor(np.ascontiguousarray(kp1), dtype=torch.float, device=device).unsqueeze(0)

            b0, k0, c0 = kp0.shape
            b1, k1, c1 = kp1.shape

            matches0 = torch.arange(k0, device=device).unsqueeze(0)
            matches1 = matches0.clone()

            mscores0 = (matches0 > -1).float()
            mscores1 = (matches1 > -1).float()

            pred = {
                "keypoints0": kp0,
                "keypoints1": kp1,
                "matches0": matches0,
                "matches1": matches1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
            }
            
            view1, view2 = batch
            correct_intrinsic(view1)
            correct_intrinsic(view2)
            
            data = {
                # "T_0to1": view1['camera_pose'] @ torch.inverse(view2['camera_pose']),
                "T_0to1": torch.inverse(view2['camera_pose']) @ view1['camera_pose'],
                "view0": view1,
                "view1": view2,
                "scene": view1['dataset'],
                "name": f"{view1['instance']}_{view2['instance']}"
            }

            results_i = {}
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    ransac_th=th,
                    estimator=pose_estimator,
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"]

            for k, v in results_i.items():
                results[k].append(v)
            
            
            torch.cuda.synchronize()
            arr_encode = [start_event.elapsed_time(end_event) for start_event, end_event in zip(events['start_encode'], events['end_encode'])]
            arr_decode = [start_event.elapsed_time(end_event) for start_event, end_event in zip(events['start_decode'], events['end_decode'])]
            arr_downstream_head = [start_event.elapsed_time(end_event) for start_event, end_event in zip(events['start_downstream_head'], events['end_downstream_head'])]
            arr_fastnn = [start_event.elapsed_time(end_event) for start_event, end_event in zip(events['start_fastnn'], events['end_fastnn'])]
            arr_bnn = [start_event.elapsed_time(end_event) for start_event, end_event in zip(events['start_bnn'], events['end_bnn'])]
            print(f'encode({len(arr_encode)}): {sum(arr_encode)}, decode({len(arr_decode)}): {sum(arr_decode)}, downstream_head({len(arr_downstream_head)}): {sum(arr_downstream_head)}, fastnn({len(arr_fastnn)}): {sum(arr_fastnn)}, bnn({len(arr_bnn)}): {sum(arr_bnn)}, mem: {torch.cuda.memory_reserved() / 1024**2:.2f} MB, max_mem: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB')

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        print(summaries)
