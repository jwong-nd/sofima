import boto3
from botocore.exceptions import ClientError
from enum import Enum
import json
import logging
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
    Load in base tile layout, tile names, and tile volumes
    from a cloud Zarr dataset with the following naming convention: 
    {tile_name}_X_{####}_Y_{####}_Z_{####}_CH_{###}_*.zarr
    Ex: tile_X_0000_Y_0000_Z_0000_CH_405.zarr
    X/Y/Z represents tile position and CH represents channel. 
    
    Default Tile Layout: 
    o -- x
    |
    y
    Overwrite tile_layout in a subclass for different x/y basis.
    """
    def __init__(self, 
                 cloud_storage: CloudStorage,
                 bucket: str, 
                 dataset_path: str, 
                 channel: int,
                 downsample_exp: int): 
        """
        cloud_storage: CloudStorage option 
        bucket: Name of bucket
        dataset_path: Path to directory containing zarr files within bucket
        channel: Image channel
        downsample_exp: Level in image pyramid with each level
                        downsampling the original resolution by 2**downsmaple_exp.
        """
        
        self.cloud_storage = cloud_storage
        self.bucket = bucket
        self.dataset_path = dataset_path
        self.channel = channel
        self.downsample_exp = downsample_exp

        if self.cloud_storage == CloudStorage.GCS: 
            raise NotImplementedError
        
        self.tile_df = self._read_tilenames_into_dataframe()
        self.channels: list[int] = self.tile_df['channel'].unique().tolist()
        filtered_tile_df = self.tile_df[self.tile_df['channel'] == self.channel]

        # Init tile_names
        grouped_df = filtered_tile_df.groupby(['Y', 'X'])
        sorted_grouped_df = grouped_df.apply(lambda x: x.sort_values(by=['Y', 'X'], ascending=[True, True]))
        self.tile_names: list[str] = sorted_grouped_df['tile_name'].tolist()

        # Init tile_layout
        y_shape = filtered_tile_df['Y'].nunique()
        x_shape = filtered_tile_df['X'].nunique()

        tile_id = 0
        tile_layout = np.zeros((y_shape, x_shape))
        for y in range(y_shape): 
            for x in range(x_shape): 
                tile_layout[y, x] = tile_id
                tile_id += 1
        self.tile_layout: np.ndarray = np.int16(tile_layout)

        # Init tile_volumes, tile_size_xyz
        tile_volumes, tile_size_xyz = self._load_zarr_data()
        self.tile_volumes: np.ndarray = tile_volumes
        self.tile_size_xyz: tuple[int, int, int] = tile_size_xyz

        # Init voxel sizes
        zarray_json = read_json_s3(self.bucket, 
                                   f'{dataset_path}/{self.tile_names[0]}/.zattrs')
        scale_tczyx = zarray_json['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
        self.vox_size_xyz: np.ndarray = np.array(scale_tczyx[2:][::-1])


    def _read_tilenames_into_dataframe(self) -> pd.DataFrame:
        """
        Parse and load tile paths into dataframe with following schema: 
        [tile_name: str, 
         x: int, 
         y: int, 
         z: int, 
         channel: int]
        """
        schema = {
            'tile_name': [],
            'X': [],
            'Y': [], 
            'Z': [],
            'channel': []
        }
        tile_df = pd.DataFrame(schema)
        for tile_path in list_directories_s3(self.bucket, self.dataset_path):
            tile_name = Path(tile_path).name
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
                'channel': channel_num
            }
            tile_df.loc[len(tile_df)] = new_entry
        return tile_df

    def _load_zarr_data(self) -> tuple[list[ts.TensorStore], stitch_elastic.ShapeXYZ]:
        """
        Reads Zarr dataset from input location 
        and returns list of equally-sized tensorstores
        in matching order as ZarrDataset.tile_names and tile size. 
        Tensorstores are cropped to tiles at origin to the smallest tile in the set.
        
        Throws an error if zarr tiles are not homogeneous in size.
        """

        def load_zarr(bucket: str, tile_location: str) -> ts.TensorStore:
            if self.cloud_storage == CloudStorage.S3:
                return open_zarr_s3(bucket, tile_location)
            else:  # cloud == 'gcs'
                return open_zarr_gcs(bucket, tile_location)
        
        tile_volumes = []
        min_x, min_y, min_z = np.inf, np.inf, np.inf

        # Average for QC 
        AVERAGE_TOLERANCE = 100
        cum_sum_zyx = np.array([0, 0, 0])

        for num, t_name in enumerate(self.tile_names):
            tile_location = f"{self.dataset_path}/{t_name}/{self.downsample_exp}"
            tile = load_zarr(self.bucket, tile_location)
            tile_volumes.append(tile)
            
            _, _, tz, ty, tx = tile.shape
            cum_sum_zyx += np.array([tz, ty, tx])
            avg_zyx = cum_sum_zyx / (num + 1.)
            assert abs(avg_zyx[0] - tz) < AVERAGE_TOLERANCE and \
                   abs(avg_zyx[1] - ty) < AVERAGE_TOLERANCE and \
                   abs(avg_zyx[2] - tx) < AVERAGE_TOLERANCE, \
                f'Average zyx shape is {(avg_zyx[0], avg_zyx[1], avg_zyx[2])} and' + \
                f'Shape of {t_name} is {(tz, ty, tx)}.'

            min_x, min_y, min_z = int(np.minimum(min_x, tx)), \
                                int(np.minimum(min_y, ty)), \
                                int(np.minimum(min_z, tz))
        tile_size_xyz = min_x, min_y, min_z

        # Standardize size of tile volumes
        for i, tile_vol in enumerate(tile_volumes):
            tile_volumes[i] = tile_vol[:, :, :min_z, :min_y, :min_x]
        
        return tile_volumes, tile_size_xyz

class SmartSpimDataset(ZarrDataset):
    """
    Dataloading customized for SmartSPIM microscope datasets.
    """
    def __init__(self, 
                 cloud_storage: CloudStorage,
                 bucket: str, 
                 dataset_path: str, 
                 channel: int,
                 downsample_exp: int):
        """
        Same parameters as parent class.
        """
        super().__init__(cloud_storage, 
                         bucket, 
                         dataset_path, 
                         channel,
                         downsample_exp)
        
        # TODO: Update tile_layout
        # TODO: Add any prerequisite tile transforms
        # to match orientation expected by processing dispim datasets thus far. 


class ExaSpimDataset(ZarrDataset):
    """
    Dataloading customized for ExaSPIM microscope datasets.
    """
    def __init__(self, 
                 cloud_storage: CloudStorage,
                 bucket: str, 
                 dataset_path: str, 
                 channel: int,
                 downsample_exp: int):
        """
        Same parameters as parent class.
        """
        super().__init__(cloud_storage, 
                         bucket, 
                         dataset_path, 
                         channel,
                         downsample_exp)
        
        # Modify tile_layout, basis is mirrored
        self.tile_layout = np.fliplr(self.tile_layout)

        # TODO: Add any prerequisite tile transforms
        # to match orientation expected by processing dispim datasets thus far. 

class DiSpimDataset(ZarrDataset):
    """
    Dataloading customized for DiSPIM microscope datasets.
    """
    def __init__(self, 
                 cloud_storage: CloudStorage,
                 bucket: str, 
                 dataset_path: str, 
                 channel: int, 
                 downsample_exp: int,
                 camera_num: int = 1, 
                 axis_flip: bool = True):
        """
        Added Parameters: 
        camera_num: 0 or 1, input to deskewing. 
        axis_flip: Tile naming convention: x -> y and y -> x. 
        """
        super().__init__(cloud_storage, 
                         bucket, 
                         dataset_path, 
                         channel,
                         downsample_exp)

        self.camera_num = camera_num
        self.axis_flip = axis_flip

        # Filter tile_names, tile_volumes by camera_num
        # Only applies if both camera streams are collected 
        match = re.search(r'(cam|CAM)', self.tile_names[0])
        if match: 
            self.tile_volumes = self.tile_volumes[camera_num::2]
            self.tile_names = self.tile_names[camera_num::2]

        # Modify tile_layout, basis dependent on camera/axis_flip: 
        if camera_num == 1:
            self.tile_layout = np.flipud(np.fliplr(self.tile_layout))
        if axis_flip: 
            self.tile_layout = np.transpose(self.tile_layout)

        # Store deskewing angle
        self.theta = 45
        if camera_num == 1: 
            self.theta = -45

def list_directories_s3(bucket_name: str, 
                        directory: str):
    if directory.endswith('/') is False:
        directory = directory + '/'

    client = boto3.client('s3')
    result = client.list_objects(Bucket=bucket_name, Prefix=directory, Delimiter='/')

    files: list[str] = []
    for o in result.get('CommonPrefixes'):
        files.append(o.get('Prefix'))

    return files


def read_json_s3(bucket_name: str,
                 json_path: str) -> dict:

    s3 = boto3.resource("s3")
    content_object = s3.Object(bucket_name, json_path)

    try:
        file_content = content_object.get()["Body"].read().decode("utf-8")
        json_content = json.loads(file_content)
    except ClientError as ex:
        if ex.response["Error"]["Code"] == "NoSuchKey":
            json_content = {}
            print(
                f"An error occurred when trying to read json file from {json_path}"
            )
        else:
            raise

    return json_content


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