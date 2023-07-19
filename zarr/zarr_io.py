import boto3
from enum import Enum
import numpy as np
from pathlib import Path
import pandas as pd
import re
import tensorstore as ts

from sofima import stitch_elastic

class CloudStorage(Enum):
    """
    Documented Cloud Storage Options
    """
    S3 = 1
    GCS = 2

class ZarrDataset:
    """
    Parameters for locating Zarr dataset living on the cloud.
    Args:
    cloud_storage: CloudStorage option 
    bucket: Name of bucket
    dataset_path: Path to directory containing zarr files within bucket
    tile_names: List of zarr tiles to include in dataset. 
                Order of tile_names defines an index that 
                is expected to be used in tile_layout.
    tile_layout: 2D array of indices that defines relative position of tiles.
    downsample_exp: Level in image pyramid with each level
                    downsampling the original resolution by 2**downsmaple_exp.
    """
    # TODO: Update this documentation

    def __init__(self, 
                 cloud_storage: CloudStorage,
                 bucket: str, 
                 dataset_path: str, 
                 downsample_exp: int, 
                 camera_num: int = 1, 
                 axis_flip: bool = True): 
        
        self.cloud_storage = cloud_storage
        self.bucket = bucket
        self.dataset_path = dataset_path
        self.downsample_exp = downsample_exp
        
        if self.cloud_storage == CloudStorage.GCS: 
            # TODO: Not implemented yet error
            pass

        # Parse and load tile paths into dataframe
        schema = {
            'tile_name': [], 
            'X': [], 
            'Y': [],
            'Z': [],
            'channel': []
        }
        tile_df = pd.Dataframe(schema)
        for tile_path in list_directories_s3(bucket):
            tile_name = Path(tile_path).stem
            if tile_name == '.zgroup':
                continue

            match = re.search(r'X_(\d+)', tile_path)
            x_pos = int(match.group(1))

            match = re.search(r'Y_(\d+)', tile_path)
            y_pos = int(match.group(1))

            match = re.search(r'Z_(\d+)', tile_path)
            z_pos = int(match.group(1))

            match = re.search(r'(ch|CH)_(\d+)', tile_path)
            channel_num = int(match.group(2))

            new_entry = {
                'tile_name': tile_name,
                'X': x_pos,
                'Y': y_pos,
                'Z': z_pos,
                'channel_num': channel_num
            }
            df = df.append(new_entry, ignore_index=True)
        
        # Init tile_names
        grouped_df = tile_df.groupby(['Y', 'X'])
        sorted_grouped_df = grouped_df.apply(lambda x: x.sort_values(by=['Y', 'X'], ascending=[True, True]))
        self.tile_names: list[str] = sorted_grouped_df['tile_name'].tolist()

        # Init tile_layout
        y_shape = len(grouped_df['Y'].nunique())
        x_shape = len(grouped_df['X'].nunique())

        tile_id = 0
        tile_layout = np.zeros((y_shape, x_shape))
        for y in range(y_shape): 
            for x in range(x_shape): 
                tile_layout[y, x] = tile_id
                tile_id += 1
        if camera_num == 1:
            tile_layout = np.flipud(np.fliplr(tile_layout))
        if axis_flip: 
            tile_layout = np.transpose(tile_layout)
        self.tile_layout: np.ndarray = tile_layout

        # Init tile_volumes, tile_size_xyz
        tile_volumes, tile_size_xyz = self._load_zarr_data()
        self.tile_volumes: np.ndarray = tile_volumes
        self.tile_size_xyz: tuple[int, int, int] = tile_size_xyz
        
        # Init tile_mesh
        # Initalization to coarse registration respecting deskewing
        # tile_mesh holds relative offsets, therefore, only holds z positions. 
        theta = 45
        if camera_num == 1: 
            theta = -45
        deskew_factor = np.tan(np.deg2rad(theta))
        deskew = np.array([[1, 0, 0], [0, 1, 0], [deskew_factor, 0, 1]])

        tile_mesh = np.zeros((3, tile_layout.shape[0], tile_layout.shape[1]))
        mx, my, mz = self.tile_size_xyz
        for y in range(y_shape): 
            for x in range(x_shape):
                tile_position_xyz = np.array([x * mx, y * my, 0])
                deskewed_position_xyz = deskew @ tile_position_xyz
                tile_mesh[:, y, x] = np.array([0, 0, deskewed_position_xyz[2]])
        self.tile_mesh: np.ndarray = tile_mesh 

    def _load_zarr_data(self) -> tuple[list[ts.TensorStore], stitch_elastic.ShapeXYZ]:
        """
        Reads Zarr dataset from input location 
        and returns list of equally-sized tensorstores
        in matching order as ZarrDataset.tile_names and tile size. 
        Tensorstores are cropped to tiles at origin to the smallest tile in the set.
        """
        
        def load_zarr(bucket: str, tile_location: str) -> ts.TensorStore:
            if self.cloud_storage == CloudStorage.S3:
                return open_zarr_s3(bucket, tile_location)
            else:  # cloud == 'gcs'
                return open_zarr_gcs(bucket, tile_location)
        tile_volumes = []
        min_x, min_y, min_z = np.inf, np.inf, np.inf
        for t_name in self.tile_names:
            tile_location = f"{self.dataset_path}/{t_name}/{self.downsample_exp}"
            tile = load_zarr(self.bucket, tile_location)
            tile_volumes.append(tile)
            
            _, _, tz, ty, tx = tile.shape
            min_x, min_y, min_z = int(np.minimum(min_x, tx)), \
                                int(np.minimum(min_y, ty)), \
                                int(np.minimum(min_z, tz))
        tile_size_xyz = min_x, min_y, min_z

        # Standardize size of tile volumes
        for i, tile_vol in enumerate(tile_volumes):
            tile_volumes[i] = tile_vol[:, :, :min_z, :min_y, :min_x]
            
        return tile_volumes, tile_size_xyz


def list_directories_s3(bucket_name: str): 
    files: list[str] = []
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects")
    result = paginator.paginate(Bucket=bucket_name, Delimiter="/")
    for prefix in result.search("CommonPrefixes"):
        files.append(prefix.get("Prefix").strip("/"))
    return files


def list_directories_gcp(): 
    # TODO: Error, not supported/implemented yet. 
    pass


def open_zarr_gcs(bucket: str, path: str) -> ts.TensorStore:
    return ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'gcs',
            'bucket': bucket,
        },
        'path': path,
    }).result()


def open_zarr_s3(bucket: str, path: str) -> ts.TensorStore: 
    return ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'http',
            'base_url': f'https://{bucket}.s3.us-west-2.amazonaws.com/{path}',
        },
    }).result()


def load_zarr_data(params: ZarrDataset
                   ) -> tuple[list[ts.TensorStore], stitch_elastic.ShapeXYZ]:
    """
    Reads Zarr dataset from input location 
    and returns list of equally-sized tensorstores
    in matching order as ZarrDataset.tile_names and tile size. 
    Tensorstores are cropped to tiles at origin to the smallest tile in the set.
    """
    
    def load_zarr(bucket: str, tile_location: str) -> ts.TensorStore:
        if params.cloud_storage == CloudStorage.S3:
            return open_zarr_s3(bucket, tile_location)
        else:  # cloud == 'gcs'
            return open_zarr_gcs(bucket, tile_location)
    tile_volumes = []
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    for t_name in params.tile_names:
        tile_location = f"{params.dataset_path}/{t_name}/{params.downsample_exp}"
        tile = load_zarr(params.bucket, tile_location)
        tile_volumes.append(tile)
        
        _, _, tz, ty, tx = tile.shape
        min_x, min_y, min_z = int(np.minimum(min_x, tx)), \
                              int(np.minimum(min_y, ty)), \
                              int(np.minimum(min_z, tz))
    tile_size_xyz = min_x, min_y, min_z

    # Standardize size of tile volumes
    for i, tile_vol in enumerate(tile_volumes):
        tile_volumes[i] = tile_vol[:, :, :min_z, :min_y, :min_x]
        
    return tile_volumes, tile_size_xyz


def write_zarr(bucket: str, shape: list, path: str): 
    """ 
    Args: 
    bucket: Name of gcs cloud storage bucket 
    shape: 5D vector in tczyx order, ex: [1, 1, 3551, 576, 576]
    path: Output path inside bucket
    """
    
    return ts.open({
        'driver': 'zarr', 
        'dtype': 'uint16',
        'kvstore' : {
            'driver': 'gcs', 
            'bucket': bucket,
        }, 
        'create': True,
        'delete_existing': True, 
        'path': path, 
        'metadata': {
        'chunks': [1, 1, 128, 256, 256],
        'compressor': {
          'blocksize': 0,
          'clevel': 1,
          'cname': 'zstd',
          'id': 'blosc',
          'shuffle': 1,
        },
        'dimension_separator': '/',
        'dtype': '<u2',
        'fill_value': 0,
        'filters': None,
        'order': 'C',
        'shape': shape,  
        'zarr_format': 2
        }
    }).result()