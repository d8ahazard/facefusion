import hashlib
from typing import List, Optional, Tuple

import numpy

from facefusion import state_manager
from facefusion.typing import Face, FaceSet, FaceStore, VisionFrame

FACE_STORE : FaceStore =\
{
	'static_faces': {},
	'reference_faces': {}
}


FACE_STORE_2: FaceStore = \
	{
		'static_faces': {},
		'reference_faces': {}
	}


def get_face_store() -> FaceStore:
	return FACE_STORE


def get_static_faces(vision_frame: VisionFrame, dict_2=False) -> Optional[List[Face]]:
	frame_hash = create_frame_hash(vision_frame)
	if dict_2:
		if frame_hash in FACE_STORE_2['static_faces']:
			return FACE_STORE_2['static_faces'][frame_hash]
		return None
	if frame_hash in FACE_STORE['static_faces']:
		return FACE_STORE['static_faces'][frame_hash]
	return None


def set_static_faces(vision_frame: VisionFrame, faces: List[Face], dict_2=False) -> None:
	frame_hash = create_frame_hash(vision_frame)
	if frame_hash:
		if dict_2:
			FACE_STORE_2['static_faces'][frame_hash] = faces
			return
		FACE_STORE['static_faces'][frame_hash] = faces


def clear_static_faces() -> None:
	FACE_STORE['static_faces'] = {}
	FACE_STORE_2['static_faces'] = {}


def create_frame_hash(vision_frame : VisionFrame) -> Optional[str]:
	return hashlib.sha1(vision_frame.tobytes()).hexdigest() if numpy.any(vision_frame) else None


def get_reference_faces() -> Optional[FaceSet]:
	if FACE_STORE['reference_faces']:
		return FACE_STORE['reference_faces']
	return None


def get_reference_faces_multi() -> Tuple[Optional[FaceSet], Optional[FaceSet]]:
	set_out = {
		'reference_faces': []
	}
	set_out_2 = {
		'reference_faces': []
	}

	ref_dict = FACE_STORE['reference_faces']
	for name in ref_dict:
		for face in ref_dict[name]:
			if face:
				set_out['reference_faces'].append(face)

	ref_dict_2 = FACE_STORE_2['reference_faces']
	for name in ref_dict_2:
		for face in ref_dict_2[name]:
			if face:
				set_out_2['reference_faces'].append(face)

	return set_out, set_out_2


def append_reference_face(name: str, face: Face, dict_2=False) -> None:
	if dict_2:
		if name not in FACE_STORE_2['reference_faces']:
			FACE_STORE_2['reference_faces'][name] = []
		FACE_STORE_2['reference_faces'][name].append(face)
		return
	if name not in FACE_STORE['reference_faces']:
		FACE_STORE['reference_faces'][name] = []
	FACE_STORE['reference_faces'][name].append(face)


def delete_reference_face(name: str, face: Face, dict_2=False) -> None:
	try:
		if dict_2:
			if name in FACE_STORE_2['reference_faces']:
				FACE_STORE_2['reference_faces'][name].remove(face)
			return
		if name in FACE_STORE['reference_faces']:
			FACE_STORE['reference_faces'][name].remove(face)
	except:
		pass


def clear_reference_faces() -> None:
	FACE_STORE['reference_faces'] = {}
	FACE_STORE_2['reference_faces'] = {}
