import io
from base64 import b64decode, b64encode
from typing import Any, Dict, Union

import googleapiclient
import numpy as np
import PIL
from google.api_core.client_options import ClientOptions


def numpy_to_b64(
    rawdata: np.ndarray, scale_factor: int = 255, format: str = "PNG"
) -> bytes:
    pil_img = PIL.Image.fromarray((rawdata * scale_factor).astype("uint8"))
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    b64str = b64encode(buffer.getvalue())
    return b64str


def b64_to_numpy(b64str: str) -> np.ndarray:
    pil_img = PIL.Image.open(io.BytesIO(b64decode(b64str)))
    return np.array(pil_img)


def predict_json(
    project: str,
    region: str,
    model: str,
    instances: Dict[str, Any],
    version: Union[str, None] = None,
) -> np.ndarray:
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build("ml", "v1", client_options=client_options)
    name = "projects/{}/models/{}".format(project, model)

    if version is not None:
        name += "/versions/{}".format(version)

    response = (
        service.projects().predict(name=name, body={"instances": instances}).execute()
    )

    if "error" in response:
        raise RuntimeError(response["error"])

    predictions: np.ndarray = response["predictions"]
    return predictions
