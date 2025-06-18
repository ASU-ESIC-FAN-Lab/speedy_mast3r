import os.path as osp
import numpy as np
import PIL

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from .utils import cropping as cropping

class RelPoseMegaDepth1500(BaseStereoViewDataset):
    def __init__(self, root, pairsfile, *args, **kwargs):
        self.ROOT = root
        super().__init__(*args, **kwargs)
        self.metadata = dict(np.load(osp.join(self.ROOT, f'megadepth_meta_test.npz'), allow_pickle=True))
        with open(osp.join(self.ROOT, f'megadepth_test_pairs.txt'), 'r') as f:
            self.scenes = f.readlines()
        self.load_depth = False
        
    def __len__(self):
        return len(self.scenes)
    
    def _crop_resize_if_necessary(self, image, intrinsics, resolution, rng=None, info=None):
        """ 
        siyan: this function can change the camera center, but the corresponding pose does not transform accordingly...
        """
        if not isinstance(image, PIL.Image.Image): 
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        # image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        image, intrinsics = cropping.crop_image(image, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        # image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)
        image, intrinsics = cropping.rescale_image(image, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        # image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        image, intrinsics2 = cropping.crop_image(image, intrinsics, crop_bbox)

        return image, intrinsics2

    
    def _get_views(self, idx, resolution,  rng):
        """
        load data for megadepth_validation views
        """
        # load metadata
        views = []
        image_idx1, image_idx2 = self.scenes[idx].strip().split(' ')
        view_idxs = [image_idx1, image_idx2]
        for view_idx in view_idxs:
            input_image_filename = osp.join(self.ROOT, view_idx)
            # load rgb images
            input_rgb_image = imread_cv2(input_image_filename)
            # load metadata
            intrinsics = np.float32(self.metadata[view_idx].item()['intrinsic'])
            camera_pose = np.linalg.inv(np.float32(self.metadata[view_idx].item()['pose']))
            # camera_pose = np.float32(self.metadata[view_idx].item()['pose'])
            # camera_pose = np.loadtxt(pose_path).astype(np.float32)

            image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, intrinsics, resolution, rng=rng, info=(self.ROOT, view_idx))
                        
            w, h = image.size
            views.append(dict(
                img=image,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=self.ROOT,
                instance=view_idx,
                depthmap=np.zeros((h, w), dtype=np.float32),  # dummy
            ))
        return views

    