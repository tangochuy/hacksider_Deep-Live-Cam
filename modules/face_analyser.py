from typing import Any, Optional, List
import insightface

import modules.globals
from modules.typing import Frame

FACE_ANALYSER: Optional[insightface.app.FaceAnalysis] = None

def get_face_analyser() -> insightface.app.FaceAnalysis:
    """
    Initializes and returns the face analyzer.
    """
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

def get_one_face(frame: Frame) -> Optional[insightface.app.Face]:
    """
    Returns the face with the smallest x-coordinate in the bounding box from the given frame.
    """
    faces = get_face_analyser().get(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except ValueError:
        return None

def get_many_faces(frame: Frame) -> Optional[List[insightface.app.Face]]:
    """
    Returns all faces detected in the given frame.
    """
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
