from arkitekt import register
import time
from cv2 import VideoCapture
from mikro.api.schema import (
    from_xarray,
    RepresentationFragment,
    ROIFragment,
    create_roi,
    InputVector,
    RoiTypeInput,
)
import xarray as xr
from typing import List, Optional
import cv2
import numpy as np
import dask.array as da

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
cam_port = 0

cam = VideoCapture(cam_port)

# reading the input using the camera


@register
def capture_image() -> RepresentationFragment:
    """Capture Frame

    Capture a frame with the webcam

    """

    result, image = cam.read()
    if result:
        return from_xarray(
            xr.DataArray(np.invert(image[:, :, 0:1]), dims=["y", "x", "c"]),
            name="Hallo",
            variety="RGB",
        )


@register
def capture_video(roi: ROIFragment, timeout: int = 2) -> RepresentationFragment:
    """Capture Video of ROI

    Capture an image with the webcam

    """

    time_start = time.time()
    frames = []
    while time.time() - time_start < timeout:
        result, image = cam.read()

        inverted = np.invert(image[:, :, 0:1])

        print(roi.vectors[0].x, roi.vectors[0].y)
        print(roi.vectors[2].x, roi.vectors[2].y)

        cropped_image = inverted[
            int(roi.vectors[0].y) : int(roi.vectors[2].y),
            int(roi.vectors[0].x) : int(roi.vectors[2].x),
            :,
        ]

        if result:
            frames.append(np.invert(cropped_image))

    return from_xarray(
        xr.DataArray(da.stack(np.array(frames)), dims=["t", "y", "x", "c"]),
        name="Video of",
        roi_origins=[roi],
    )


@register
def blob_detection(
    image: RepresentationFragment,
    max_blobs: Optional[int] = 4,
    min_radius: int = 20,
    max_radius: int = 700,
) -> List[ROIFragment]:
    """Detect Blobs in Image

    Detect blobs in an image using the OpenCV Hough Transform
    and creates ROIs for each blob.


    """
    # Set up the detector with default parameters.

    x = image.data.sel(c=0, t=0, z=0).data.compute()
    blurred = cv2.GaussianBlur(x, (9, 9), 0)
    # Detect circles in the image using Hough transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    print(circles)

    rois = []

    # Ensure at least some circles were found
    if circles is not None:
        # Convert the circle parameters (x, y, radius) to integers
        circles = np.round(circles[0, :]).astype("int")
        i = 0
        # Loop over the circles
        for x, y, r in circles:
            if i > max_blobs:
                break
            print(x, y, r)

            roi = create_roi(
                image,
                vectors=[
                    InputVector(x=x - r, y=y - r, c=0, t=0, z=0),
                    InputVector(x=x + r, y=y - r, c=0, t=0, z=0),
                    InputVector(x=x + r, y=y + r, c=0, t=0, z=0),
                    InputVector(x=x - r, y=y + r, c=0, t=0, z=0),
                ],
                label="Blob",
                type=RoiTypeInput.RECTANGLE,
            )

            rois.append(roi)
            i += 1

    return rois
