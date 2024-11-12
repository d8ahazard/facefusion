from typing import List, Optional, Tuple

import gradio
import numpy as np
from gradio_rangeslider import RangeSlider

import facefusion.choices
from facefusion import state_manager, wording
from facefusion.common_helper import calc_float_step, calc_int_step
from facefusion.face_analyser import get_many_faces
from facefusion.face_selector import sort_and_filter_faces
from facefusion.face_store import clear_reference_faces, clear_static_faces, append_reference_face, \
	delete_reference_face
from facefusion.filesystem import is_image, is_video
from facefusion.typing import FaceSelectorMode, FaceSelectorOrder, Gender, Race, VisionFrame
from facefusion.uis.core import get_ui_component, get_ui_components, register_ui_component
from facefusion.uis.typing import ComponentOptions
from facefusion.uis.ui_helper import convert_str_none
from facefusion.vision import get_video_frame, normalize_frame_color, read_static_image

FACE_SELECTOR_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_ORDER_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_GENDER_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_RACE_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SELECTOR_AGE_RANGE_SLIDER : Optional[RangeSlider] = None
REFERENCE_FACE_POSITION_GALLERY : Optional[gradio.Gallery] = None
REFERENCE_FACE_DISTANCE_SLIDER : Optional[gradio.Slider] = None
ADD_REFERENCE_FACE_BUTTON : Optional[gradio.Button] = None
REMOVE_REFERENCE_FACE_BUTTON : Optional[gradio.Button] = None
REFERENCE_FACES_SELECTION_GALLERY : Optional[gradio.Gallery] = None
ADD_REFERENCE_FACE_BUTTON_2 : Optional[gradio.Button] = None
REMOVE_REFERENCE_FACE_BUTTON_2 : Optional[gradio.Button] = None
REFERENCE_FACES_SELECTION_GALLERY_2 : Optional[gradio.Gallery] = None

current_reference_faces = []
current_reference_frames = []

current_selected_faces = []
current_selected_faces_2 = []
selector_face_index = -1
selected_face_index = -1
selected_face_index_2 = -1

def render() -> None:
	global FACE_SELECTOR_MODE_DROPDOWN
	global FACE_SELECTOR_ORDER_DROPDOWN
	global FACE_SELECTOR_GENDER_DROPDOWN
	global FACE_SELECTOR_RACE_DROPDOWN
	global FACE_SELECTOR_AGE_RANGE_SLIDER
	global REFERENCE_FACE_POSITION_GALLERY
	global ADD_REFERENCE_FACE_BUTTON
	global REMOVE_REFERENCE_FACE_BUTTON
	global REFERENCE_FACES_SELECTION_GALLERY
	global REFERENCE_FACES_SELECTION_GALLERY_2
	global ADD_REFERENCE_FACE_BUTTON_2
	global REMOVE_REFERENCE_FACE_BUTTON_2
	global REFERENCE_FACE_DISTANCE_SLIDER

	reference_face_gallery_options : ComponentOptions =\
	{
		'label': wording.get('uis.reference_face_gallery'),
		'object_fit': 'cover',
		'columns': 8,
		'allow_preview': False,
		'interactive': False,
		'visible': 'reference' in state_manager.get_item('face_selector_mode')
	}
	if is_image(state_manager.get_item('target_path')):
		reference_frame = read_static_image(state_manager.get_item('target_path'))
		reference_face_gallery_options['value'] = extract_gallery_frames(reference_frame)
	if is_video(state_manager.get_item('target_path')):
		reference_frame = get_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
		reference_face_gallery_options['value'] = extract_gallery_frames(reference_frame)
	FACE_SELECTOR_MODE_DROPDOWN = gradio.Dropdown(
		label = wording.get('uis.face_selector_mode_dropdown'),
		choices = facefusion.choices.face_selector_modes,
		value = state_manager.get_item('face_selector_mode')
	)

	with gradio.Row():
		with gradio.Column(scale=0, min_width="33"):
			ADD_REFERENCE_FACE_BUTTON = gradio.Button(
				value="+1",
				elem_id='ff_add_reference_face_button'
			)
			ADD_REFERENCE_FACE_BUTTON_2 = gradio.Button(
				value="+2",
				elem_id='ff_add_reference_face_button'
			)
		with gradio.Column():
			REFERENCE_FACE_POSITION_GALLERY = gradio.Gallery(**reference_face_gallery_options,
															 elem_id='ff_reference_face_position_gallery')
	with gradio.Row():
		REMOVE_REFERENCE_FACE_BUTTON = gradio.Button(
			value="-",
			variant='secondary',
			elem_id='ff_remove_reference_faces_button',
			elem_classes=['remove_reference_faces_button']
		)
		REFERENCE_FACES_SELECTION_GALLERY = gradio.Gallery(
			label="Selected Faces (Source 1)",
			interactive=False,
			object_fit='cover',
			columns=8,
			allow_preview=False,
			visible='reference' in state_manager.get_item('face_selector_mode'),
			elem_id='ff_reference_faces_selection_gallery'
		)
	with gradio.Row():
		REMOVE_REFERENCE_FACE_BUTTON_2 = gradio.Button(
			value="-",
			variant='secondary',
			elem_id='ff_remove_reference_faces_button_2',
			elem_classes=['remove_reference_faces_button']
		)
		REFERENCE_FACES_SELECTION_GALLERY_2 = gradio.Gallery(
			label="Selected Faces (Source 2)",
			interactive=False,
			object_fit='cover',
			columns=8,
			allow_preview=False,
			visible='reference' in state_manager.get_item('face_selector_mode'),
			elem_id='ff_reference_faces_selection_gallery'
		)
	with gradio.Group():
		with gradio.Row():
			FACE_SELECTOR_ORDER_DROPDOWN = gradio.Dropdown(
				label = wording.get('uis.face_selector_order_dropdown'),
				choices = facefusion.choices.face_selector_orders,
				value = state_manager.get_item('face_selector_order')
			)
			FACE_SELECTOR_GENDER_DROPDOWN = gradio.Dropdown(
				label = wording.get('uis.face_selector_gender_dropdown'),
				choices = [ 'none' ] + facefusion.choices.face_selector_genders,
				value = state_manager.get_item('face_selector_gender') or 'none'
			)
			FACE_SELECTOR_RACE_DROPDOWN = gradio.Dropdown(
				label = wording.get('uis.face_selector_race_dropdown'),
				choices = ['none'] + facefusion.choices.face_selector_races,
				value = state_manager.get_item('face_selector_race') or 'none'
			)
		with gradio.Row():
			face_selector_age_start = state_manager.get_item('face_selector_age_start') or facefusion.choices.face_selector_age_range[0]
			face_selector_age_end = state_manager.get_item('face_selector_age_end') or facefusion.choices.face_selector_age_range[-1]
			FACE_SELECTOR_AGE_RANGE_SLIDER = RangeSlider(
				label = wording.get('uis.face_selector_age_range_slider'),
				minimum = facefusion.choices.face_selector_age_range[0],
				maximum = facefusion.choices.face_selector_age_range[-1],
				value = (face_selector_age_start, face_selector_age_end),
				step = calc_int_step(facefusion.choices.face_selector_age_range)
			)
	REFERENCE_FACE_DISTANCE_SLIDER = gradio.Slider(
		label = wording.get('uis.reference_face_distance_slider'),
		value = state_manager.get_item('reference_face_distance'),
		step = calc_float_step(facefusion.choices.reference_face_distance_range),
		minimum = facefusion.choices.reference_face_distance_range[0],
		maximum = facefusion.choices.reference_face_distance_range[-1],
		visible = 'reference' in state_manager.get_item('face_selector_mode')
	)
	register_ui_component('face_selector_mode_dropdown', FACE_SELECTOR_MODE_DROPDOWN)
	register_ui_component('face_selector_order_dropdown', FACE_SELECTOR_ORDER_DROPDOWN)
	register_ui_component('face_selector_gender_dropdown', FACE_SELECTOR_GENDER_DROPDOWN)
	register_ui_component('face_selector_race_dropdown', FACE_SELECTOR_RACE_DROPDOWN)
	register_ui_component('face_selector_age_range_slider', FACE_SELECTOR_AGE_RANGE_SLIDER)
	register_ui_component('reference_face_position_gallery', REFERENCE_FACE_POSITION_GALLERY)
	register_ui_component('reference_face_distance_slider', REFERENCE_FACE_DISTANCE_SLIDER)
	register_ui_component('reference_faces_selection_gallery', REFERENCE_FACES_SELECTION_GALLERY)
	register_ui_component('add_reference_face_button', ADD_REFERENCE_FACE_BUTTON)
	register_ui_component('remove_reference_faces_button', REMOVE_REFERENCE_FACE_BUTTON)
	register_ui_component('reference_faces_selection_gallery', REFERENCE_FACES_SELECTION_GALLERY)
	register_ui_component('add_reference_face_button_2', ADD_REFERENCE_FACE_BUTTON_2)


def listen() -> None:
	face_selector_elements = [ REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY_2, REFERENCE_FACE_DISTANCE_SLIDER,
							   ADD_REFERENCE_FACE_BUTTON, REMOVE_REFERENCE_FACE_BUTTON, ADD_REFERENCE_FACE_BUTTON_2, REMOVE_REFERENCE_FACE_BUTTON_2]
	FACE_SELECTOR_MODE_DROPDOWN.change(update_face_selector_mode, inputs = FACE_SELECTOR_MODE_DROPDOWN, outputs = face_selector_elements)
	FACE_SELECTOR_ORDER_DROPDOWN.change(update_face_selector_order, inputs = FACE_SELECTOR_ORDER_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	FACE_SELECTOR_GENDER_DROPDOWN.change(update_face_selector_gender, inputs = FACE_SELECTOR_GENDER_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	FACE_SELECTOR_RACE_DROPDOWN.change(update_face_selector_race, inputs = FACE_SELECTOR_RACE_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	FACE_SELECTOR_AGE_RANGE_SLIDER.release(update_face_selector_age_range, inputs = FACE_SELECTOR_AGE_RANGE_SLIDER, outputs = REFERENCE_FACE_POSITION_GALLERY)
	#REFERENCE_FACE_POSITION_GALLERY.select(clear_and_update_reference_face_position)
	REFERENCE_FACE_POSITION_GALLERY.select(update_selector_face_index, show_progress="hidden")
	REFERENCE_FACES_SELECTION_GALLERY.select(update_selected_face_index)
	REFERENCE_FACES_SELECTION_GALLERY_2.select(update_selected_face_index_2)
	REFERENCE_FACE_DISTANCE_SLIDER.release(update_reference_face_distance, inputs = REFERENCE_FACE_DISTANCE_SLIDER)

	for ui_component in get_ui_components(
	[
		'target_image',
		'target_video'
	]):
		for method in [ 'upload', 'change', 'clear' ]:
			getattr(ui_component, method)(update_reference_face_position)
			getattr(ui_component, method)(update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)

	for ui_component in get_ui_components(
	[
		'face_detector_model_dropdown',
		'face_detector_size_dropdown',
		'face_detector_angles_checkbox_group'
	]):
		ui_component.change(clear_and_update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)

	face_detector_score_slider = get_ui_component('face_detector_score_slider')
	if face_detector_score_slider:
		face_detector_score_slider.release(clear_and_update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)

	preview_frame_slider = get_ui_component('preview_frame_slider')
	preview_image = get_ui_component('preview_image')
	if preview_frame_slider:
		ADD_REFERENCE_FACE_BUTTON.click(add_reference_face,
										inputs=[REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY, preview_frame_slider],
										outputs=[REFERENCE_FACES_SELECTION_GALLERY, preview_image])
		ADD_REFERENCE_FACE_BUTTON_2.click(add_reference_face_2,
										  inputs=[REFERENCE_FACE_POSITION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY_2, preview_frame_slider],
										  outputs=[REFERENCE_FACES_SELECTION_GALLERY_2, preview_image])
		REMOVE_REFERENCE_FACE_BUTTON.click(fn=remove_reference_face,
										   inputs=[REFERENCE_FACES_SELECTION_GALLERY, preview_frame_slider],
										   outputs=[REFERENCE_FACES_SELECTION_GALLERY, preview_image])

		REMOVE_REFERENCE_FACE_BUTTON_2.click(fn=remove_reference_face_2,
											 inputs=[REFERENCE_FACES_SELECTION_GALLERY_2, preview_frame_slider],
											 outputs=[REFERENCE_FACES_SELECTION_GALLERY_2, preview_image])
		preview_frame_slider.release(update_reference_frame_number, inputs = preview_frame_slider)
		preview_frame_slider.release(update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)


def update_face_selector_mode(face_selector_mode : FaceSelectorMode) -> Tuple[gradio.Gallery, gradio.Gallery, gradio.Gallery, gradio.Slider]:
	state_manager.set_item('face_selector_mode', face_selector_mode)
	show_ref = 'reference' in face_selector_mode

	return (gradio.Gallery(visible = show_ref), gradio.Gallery(visible = show_ref), gradio.Gallery(visible = show_ref),
			gradio.Slider(visible = not show_ref), gradio.Button(visible = show_ref), gradio.Button(visible = show_ref),
			gradio.Button(visible = show_ref), gradio.Button(visible = show_ref))


def update_face_selector_order(face_analyser_order : FaceSelectorOrder) -> gradio.Gallery:
	state_manager.set_item('face_selector_order', convert_str_none(face_analyser_order))
	return update_reference_position_gallery()


def update_face_selector_gender(face_selector_gender : Gender) -> gradio.Gallery:
	state_manager.set_item('face_selector_gender', convert_str_none(face_selector_gender))
	return update_reference_position_gallery()


def update_face_selector_race(face_selector_race : Race) -> gradio.Gallery:
	state_manager.set_item('face_selector_race', convert_str_none(face_selector_race))
	return update_reference_position_gallery()


def update_face_selector_age_range(face_selector_age_range : Tuple[float, float]) -> gradio.Gallery:
	face_selector_age_start, face_selector_age_end = face_selector_age_range
	state_manager.set_item('face_selector_age_start', int(face_selector_age_start))
	state_manager.set_item('face_selector_age_end', int(face_selector_age_end))
	return update_reference_position_gallery()


def update_selector_face_index(event: gradio.SelectData) -> None:
	global selector_face_index
	print("Index changed...")
	selector_face_index = event.index


def update_selected_face_index(event: gradio.SelectData) -> None:
	global selected_face_index
	print("Index changed...")
	selected_face_index = event.index


def update_selected_face_index_2(event: gradio.SelectData) -> None:
	global selected_face_index_2
	print("Index 23 changed...")
	selected_face_index_2 = event.index


def clear_selected_faces() -> None:
	global current_selected_faces, current_selected_faces_2, selected_face_index, selected_face_index_2
	current_selected_faces = []
	current_selected_faces_2 = []
	selected_face_index = -1
	selected_face_index_2 = -1
	state_manager.set_item('reference_face_dict', {})
	state_manager.set_item('reference_face_dict_2', {})


def clear_and_update_reference_face_position(event : gradio.SelectData) -> gradio.Gallery:
	clear_reference_faces()
	clear_static_faces()
	update_reference_face_position(event.index)
	return update_reference_position_gallery()


def add_reference_face(src_gallery, dest_gallery, reference_frame_number) -> Tuple[gradio.Gallery, gradio.Image]:
	global selected_face_index
	global current_selected_faces
	global current_reference_faces
	global selector_face_index
	dest_items = [item[0] for item in dest_gallery] if dest_gallery is not None and any(dest_gallery) else []
	if src_gallery is not None and any(src_gallery):
		# If the number of items in gallery is less than selected_face_index, then the selected_face_index is invalid
		if len(src_gallery) <= selector_face_index:
			selector_face_index = -1
			print("Invalid index")
			return dest_gallery, gradio.update()
		selected_item = src_gallery[selector_face_index]
		face_data = current_reference_faces[selector_face_index]
		reference_face_dict = state_manager.get_item('reference_face_dict')
		if reference_face_dict is None:
			reference_face_dict = {}
		if reference_frame_number not in reference_face_dict:
			reference_face_dict[reference_frame_number] = []
		found = False
		for existing_face_data in reference_face_dict[reference_frame_number]:
			if np.array_equal(face_data, existing_face_data):
				found = True
				break

		if not found:
			reference_face_dict[reference_frame_number].append(face_data)
			current_selected_faces.append(face_data)
			dest_items.append(selected_item[0])
			append_reference_face('reference_face', face_data)
		state_manager.set_item('reference_face_dict', reference_face_dict)

		from facefusion.uis.components.preview import update_preview_image

		out_preview = update_preview_image(reference_frame_number)
		return gradio.Gallery(value=dest_items), out_preview
	else:
		return gradio.Gallery(value=dest_items), gradio.update()


def remove_reference_face(gallery: gradio.Gallery, preview_frame_number) -> Tuple[gradio.Gallery, gradio.Image]:
	global selected_face_index, current_selected_faces, current_reference_faces
	if len(gallery) <= selected_face_index or len(current_selected_faces) <= selected_face_index:
		selected_face_index = -1
		print("Invalid index")
		return gradio.update(), gradio.update()

	# Remove the selected item from the gallery
	new_items = []
	gallery_index = 0
	for item in gallery:
		if gallery_index != selected_face_index:
			new_items.append(item[0])
		gallery_index += 1
	global_reference_faces = state_manager.get_item('reference_face_dict')
	if global_reference_faces is None:
		global_reference_faces = {}
	face_to_remove = current_selected_faces[selected_face_index]
	found = False
	for frame_no, faces in global_reference_faces.items():
		cleaned_faces = []
		for existing_face_data in faces:
			if np.array_equal(face_to_remove, existing_face_data):
				print("Found face to remove")
				found = True
				continue
			cleaned_faces.append(existing_face_data)
		global_reference_faces[frame_no] = cleaned_faces
		if found:
			delete_reference_face('reference_face', face_to_remove)
			break

	state_manager.set_item('reference_face_dict', global_reference_faces)
	current_selected_faces.pop(selected_face_index)
	from facefusion.uis.components.preview import update_preview_image
	preview_image = update_preview_image(preview_frame_number)
	return gradio.update(value=new_items), preview_image


def add_reference_face_2(src_gallery, dest_gallery, reference_frame_number) -> Tuple[gradio.Gallery, gradio.Image]:
	global selector_face_index, current_selected_faces_2, current_reference_faces
	dest_items = [item[0] for item in dest_gallery] if dest_gallery is not None else []
	if src_gallery is not None and any(src_gallery):
		# If the number of items in gallery is less than selected_face_index, then the selected_face_index is invalid
		if len(src_gallery) <= selector_face_index:
			selector_face_index = -1
			print("Invalid index")
			return gradio.update(), gradio.update(), gradio.update()
		selected_item = src_gallery[selector_face_index]
		face_data = current_reference_faces[selector_face_index]
		reference_face_dict_2 = state_manager.get_item('reference_face_dict_2')
		if reference_face_dict_2 is None:
			reference_face_dict_2 = {}
		if reference_frame_number not in reference_face_dict_2:
			reference_face_dict_2[reference_frame_number] = []
		found = False
		for existing_face_data in reference_face_dict_2[reference_frame_number]:
			if np.array_equal(face_data, existing_face_data):
				found = True
				break

		if not found:
			reference_face_dict_2[reference_frame_number].append(face_data)
			current_selected_faces_2.append(face_data)
			dest_items.append(selected_item[0])
			append_reference_face('reference_face', face_data, dict_2=True)
		state_manager.set_item('reference_face_dict_2', reference_face_dict_2)
		from facefusion.uis.components.preview import update_preview_image

		out_preview = update_preview_image(reference_frame_number)
		return gradio.Gallery(value=dest_items), out_preview
	else:
		return gradio.Gallery(value=dest_items), gradio.update()


def remove_reference_face_2(gallery: gradio.Gallery, preview_frame_number) -> Tuple[gradio.Gallery, gradio.Image]:
	global selected_face_index_2, current_selected_faces_2
	if len(gallery) <= selected_face_index_2 or len(current_selected_faces) <= selected_face_index_2:
		selected_face_index_2 = -1
		print("Invalid index")
		return gradio.update(), gradio.update()

	# Remove the selected item from the gallery
	new_items = []
	gallery_index = 0
	for item in gallery:
		if gallery_index != selected_face_index_2:
			new_items.append(item[0])
		gallery_index += 1

	global_reference_faces = state_manager.get_item('reference_face_dict_2')
	if global_reference_faces is None:
		global_reference_faces = {}
	face_to_remove = current_selected_faces[selected_face_index_2]
	found = False
	for frame_no, faces in global_reference_faces.items():
		cleaned_faces = []
		for existing_face_data in faces:
			if np.array_equal(face_to_remove, existing_face_data):
				print("Found face to remove")
				found = True
				continue
			cleaned_faces.append(existing_face_data)
		global_reference_faces[frame_no] = cleaned_faces
		if found:
			delete_reference_face('reference_face', face_to_remove, dict_2=True)
			break

	state_manager.set_item('reference_face_dict_2', global_reference_faces)
	current_selected_faces_2.pop(selected_face_index_2)
	from facefusion.uis.components.preview import update_preview_image
	preview_image = update_preview_image(preview_frame_number)
	return gradio.update(value=new_items), preview_image


def update_reference_face_position(reference_face_position : int = 0) -> None:
	state_manager.set_item('reference_face_position', reference_face_position)


def update_reference_face_distance(reference_face_distance : float) -> None:
	state_manager.set_item('reference_face_distance', reference_face_distance)


def update_reference_frame_number(reference_frame_number : int) -> None:
	state_manager.set_item('reference_frame_number', reference_frame_number)


# TODO: Fix the output of this
def clear_and_update_reference_position_gallery() -> gradio.Gallery:
	clear_reference_faces()
	clear_static_faces()
	clear_selected_faces()
	return update_reference_position_gallery()


def update_reference_position_gallery() -> gradio.Gallery:
	gallery_vision_frames = []
	selection_gallery = gradio.update()
	selection_gallery_2 = gradio.update()
	if is_image(state_manager.get_item('target_path')):
		temp_vision_frame = read_static_image(state_manager.get_item('target_path'))
		gallery_vision_frames = extract_gallery_frames(temp_vision_frame)
	elif is_video(state_manager.get_item('target_path')):
		temp_vision_frame = get_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
		gallery_vision_frames = extract_gallery_frames(temp_vision_frame)
	else:
		selection_gallery = gradio.update(value=None)
		selection_gallery_2 = gradio.update(value=None)
		state_manager.set_item('reference_face_dict', {})
		global current_selected_faces
		current_selected_faces = []
	if gallery_vision_frames:
		return gradio.Gallery(value = gallery_vision_frames)
	return gradio.Gallery(value = None)


# def update_reference_frame_number_and_gallery(reference_frame_number) -> Tuple[gradio.update, gradio.update]:
# 	gallery_frames = []
# 	facefusion.globals.reference_frame_number = reference_frame_number
# 	selection_gallery = gradio.update()
# 	selection_gallery_2 = gradio.update()
# 	if is_image(facefusion.globals.target_path):
# 		reference_frame = read_static_image(facefusion.globals.target_path)
# 		gallery_frames = extract_gallery_frames(reference_frame)
# 	elif is_video(facefusion.globals.target_path):
# 		reference_frame = get_video_frame(facefusion.globals.target_path, facefusion.globals.reference_frame_number)
# 		gallery_frames = extract_gallery_frames(reference_frame)
# 	else:
# 		selection_gallery = gradio.update(value=None)
# 		selection_gallery_2 = gradio.update(value=None)
# 		facefusion.globals.reference_face_dict = {}
# 		global current_selected_faces
# 		current_selected_faces = []
# 	if gallery_frames:
# 		return gradio.update(value=reference_frame_number), gradio.update(value=gallery_frames), selection_gallery, selection_gallery_2
# 	return gradio.update(value=reference_frame_number), gradio.update(value=None), selection_gallery, selection_gallery_2


def extract_gallery_frames(temp_vision_frame : VisionFrame) -> List[VisionFrame]:
	gallery_vision_frames = []
	faces = sort_and_filter_faces(get_many_faces([ temp_vision_frame ]))
	global current_reference_faces
	global current_reference_frames
	current_reference_faces = faces
	for face in faces:
		start_x, start_y, end_x, end_y = map(int, face.bounding_box)
		padding_x = int((end_x - start_x) * 0.25)
		padding_y = int((end_y - start_y) * 0.25)
		start_x = max(0, start_x - padding_x)
		start_y = max(0, start_y - padding_y)
		end_x = max(0, end_x + padding_x)
		end_y = max(0, end_y + padding_y)
		crop_vision_frame = temp_vision_frame[start_y:end_y, start_x:end_x]
		crop_vision_frame = normalize_frame_color(crop_vision_frame)
		gallery_vision_frames.append(crop_vision_frame)
	current_reference_frames = gallery_vision_frames
	return gallery_vision_frames
