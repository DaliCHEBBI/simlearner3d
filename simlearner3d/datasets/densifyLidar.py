import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pdal
import rasterio



def mask_raster_with_nodata(raster_to_mask: Path, mask: np.array, output_tif: Path):
    """Replace data in raster_to_mask by "no_data_value" where mask=0
    (usual masks are occupancy maps for which 1 indicates that there are data)

    Args:
        raster_to_mask (Path): Path to the raster on which to replace data
        mask (np.array): Binary Array (with the same shape as raster_to_mask data) with 1 where to keep data,
        0 where to replace them with the nodata value
        output_tif (Path): Path to the resulting file
    """
    with rasterio.Env():
        with rasterio.open(str(raster_to_mask)) as src:
            raster = src.read()
            out_meta = src.meta
            nodata = src.nodata

        raster[mask == 0] = nodata

        with rasterio.open(str(output_tif), "w", **out_meta) as dest:
            dest.write(raster)


def create_mns_map(las_file, output_tif, pixel_size, no_data_value=-9999):
    reader = pdal.Reader.text(filename=str(las_file))
    pipeline = reader.pipeline()
    top_left=(0,1024)
    nb_pixels=(1024,1024)

    lower_left = (top_left[0], top_left[1] - nb_pixels[1] * pixel_size)

    raster_tags = ["RASTER_0"]

    #pipeline |= pdal.Filter.delaunay()

    #pipeline |= pdal.Filter.voxelcentroidnearestneighbor(
    #    cell=4.0)

    """pipeline |= pdal.Filter.greedyprojection(
        multiplier=3,
        radius=2,
        min_angle=0
    )"""
    #

    pipeline |= pdal.Filter.planefit()


    pipeline |= pdal.Writer.ply(filename=output_tif)#,storage_mode="4", dataformat_id="8", forward="all")


    """pipeline |= pdal.Filter.delaunay()

    pipeline |= pdal.Filter.faceraster(
        resolution=str(pixel_size),
        origin_x=str(lower_left[0] - pixel_size / 2),
        origin_y=str(lower_left[1] + pixel_size / 2),
        width=str(nb_pixels[0]),
        height=str(nb_pixels[1]),
        tag= "RASTER_0",
    )

    pipeline |= pdal.Writer.raster(
        gdaldriver="GTiff",
        nodata=no_data_value,
        data_type="float32",
        filename=str(output_tif),
        inputs=raster_tags,
        rasters=["faceraster"],
    )"""
    pipeline.execute()
    

def mask_raster_with_nodata(raster_to_mask: Path, output_tif: Path):
    """Replace data in raster_to_mask by "no_data_value" where mask=0
    (usual masks are occupancy maps for which 1 indicates that there are data)

    Args:
        raster_to_mask (Path): Path to the raster on which to replace data
        mask (np.array): Binary Array (with the same shape as raster_to_mask data) with 1 where to keep data,
        0 where to replace them with the nodata value
        output_tif (Path): Path to the resulting file
    """
    with rasterio.Env():
        with rasterio.open(str(raster_to_mask)) as src:
            raster = src.read()
            out_meta = src.meta
            nodata = src.nodata

        #raster[mask == 0] = nodata

        with rasterio.open(str(output_tif), "w", **out_meta) as dest:
            dest.write(raster)


def rasterize_las(
    las_file: Path,
    output_tif: Path,
    pixel_size: float = 0.5,
    no_data_value=-9999,
):
    """
    Create for each class that is in config_file keys:
    - a height raster (kind of digital surface model for a single class, called mnx here)
    - if "occupancy_tif" has a value, a 2d occupancy map

    both are saved single output_tif files with one layer per class (the classes are sorted alphabetically).

    Args:
        las_file (Path): path to the las file on which to generate malt0 intrinsic metric
        config_file (Path): class weights dict in the config file (to know for which classes to generate the rasters)
        output_tif (Path): path to output height raster
        pixel_size (float, optional): size of the output rasters pixels. Defaults to 0.5.
        no_data_value (int, optional): no_data value for the output raster. Defaults to -9999.
    """

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    #with tempfile.NamedTemporaryFile(suffix=las_file.stem + "_mns.tif") as tmp_mns:
    create_mns_map(las_file, output_tif,
                    #"""tmp_mns.name""",
                      pixel_size,
                     no_data_value)
        #mask_raster_with_nodata(tmp_mns.name, output_tif)


def parse_args():
    parser = argparse.ArgumentParser("Rasterize las file")
    parser.add_argument("-i", "--input_file", type=Path, required=True, help="Path to the LAS file")
    parser.add_argument("-o", "--output_mns_file", type=Path, required=True, help="Path to the TIF output file")
    parser.add_argument("-p", "--pixel_size", type=float, required=True, help="Size of the output raster pixels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    rasterize_las(
        las_file=Path(args.input_file),
        output_tif=Path(args.output_mns_file),
        pixel_size=float(args.pixel_size)
    )
