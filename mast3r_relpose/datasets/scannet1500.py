import os.path as osp
import numpy as np
import PIL
import cv2

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from .utils import cropping as cropping

class RelPoseScanNet1500(BaseStereoViewDataset):
    def __init__(self, root, pairsfile, *args, **kwargs):
        self.ROOT = root
        super().__init__(*args, **kwargs)
        self.pairs_path = f'{self.ROOT}/{pairsfile}.npz'
        self.subfolder_mask = 'scannet_test_1500/scene{:04d}_{:02d}'
        with np.load(self.pairs_path) as data:
            self.pair_names = data['name']
        
    def __len__(self):
        return len(self.pair_names)
    
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

    def _get_views(self, idx, resolution, rng):
        scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]

        views = []

        for name in [name1, name2]: 
            
            color_path = '{}/{}/color/{}.jpg'.format(self.ROOT, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            color_image = imread_cv2(color_path)  
            color_image = cv2.resize(color_image, (640, 480))

            intrinsics_path = '{}/{}/intrinsic/intrinsic_depth.txt'.format(self.ROOT, self.subfolder_mask).format(scene_name, scene_sub_name)
            intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)[0:3,0:3]

            pose_path = '{}/{}/pose/{}.txt'.format(self.ROOT, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            camera_pose = np.loadtxt(pose_path).astype(np.float32)

            color_image, intrinsics = self._crop_resize_if_necessary(color_image, 
                                                                     intrinsics, 
                                                                     resolution, 
                                                                     rng=rng)

            view_idx_splits = color_path.split('/')

            w, h = color_image.size
            views.append(dict(
                img = color_image,
                camera_intrinsics = intrinsics,
                camera_pose = camera_pose,
                dataset = 'ScanNet1500',
                label = '_'.join(view_idx_splits[:-1]),
                instance = view_idx_splits[-1],
                depthmap=np.zeros((h, w), dtype=np.float32),  # dummy
                ))
        return views    