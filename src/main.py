import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
from datetime import datetime
import json
import pickle
import time
import numpy as np
import supervision as sv
import streamlit as  st
from tqdm import tqdm
from ultralytics import YOLO

from modules.load_models import load_model_obj, load_model_kp, load_model_line
from modules.fields import inference, projection_from_cam_params, project, inverse_projection_matrix, visualize_field_and_players

from modules.sports.annotators.soccer import draw_soccer_field, draw_players
from modules.sports.common.ball import BallTracker, BallAnnotator
from modules.sports.common.team import TeamClassifier
from modules.sports.common.view import ViewTransformer
from modules.sports.configs.soccer import SoccerFieldConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, '..', 'data')
CONFIG_DIR = os.path.join(PARENT_DIR, 'config')

PLAYER_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'football-player-detection-v9.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'football-pitch-detection-v9.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'football-ball-detection-v2.pt')
KEYPOINT_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'weights/MV_kp')
LINE_DETECTION_MODEL_PATH = os.path.join(DATA_DIR, 'weights/MV_lines')
KEYPOINT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'hrnetv2_w48.yaml')
LINE_CONFIG_PATH = os.path.join(CONFIG_DIR, 'hrnetv2_w48_l.yaml')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerFieldConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    LOCALIZATION = 'LOCALIZATION'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_soccer_field(config=CONFIG)
    radar = draw_players(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, soccer_field=radar)
    radar = draw_players(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, soccer_field=radar)
    radar = draw_players(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, soccer_field=radar)
    radar = draw_players(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, soccer_field=radar)
    return radar



def run_player_detection(source_video_path: str, device: str, player_detection_conf: float = 0.5) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False, conf=player_detection_conf)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

def run_player_tracking(source_video_path: str, device: str, player_detection_conf: float = 0.5) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False, conf=player_detection_conf)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        
        yield annotated_frame

def run_pitch_detection(source_video_path: str, device: str, keypoints_detection_conf: float = 0.5, lines_detection_conf: float = 0.5) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame

def run_field_detection(source_video_path: str, device: str, keypoints_detection_conf: float = 0.5, lines_detection_conf: float = 0.5, show_field_mode: str = 'both') -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    keypoints_detection_model = load_model_kp(KEYPOINT_CONFIG_PATH, KEYPOINT_DETECTION_MODEL_PATH)
    lines_detection_model = load_model_line(LINE_CONFIG_PATH, LINE_DETECTION_MODEL_PATH)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        try:
            h, w, _ = frame.shape
            result = inference(frame, keypoints_detection_model, lines_detection_model, keypoints_detection_conf, lines_detection_conf, h, w)
            H = projection_from_cam_params(result)
            frame = project(frame, H, show_field_mode=show_field_mode)
        
        except Exception as e:
            print(f"An error occurred during field detection: {e}")
            continue
        
        annotated_frame = frame.copy()
        yield annotated_frame

def run_ball_detection(source_video_path: str, device: str, ball_detection_conf: float = 0.5) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False, conf=ball_detection_conf)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame

def run_team_classification(source_video_path: str, device: str, player_detection_conf: float = 0.5) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False, conf=player_detection_conf)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame

def run_team_classification_and_save(
    source_video_path: str,
    device: str,
    player_detection_conf: float = 0.5,
    state_save_path: str = "team_classification_state.pkl"
):
    """
    Run team classification on a video, save the team classifier state.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        state_save_path (str): Path to save the team classifier state.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    team_classifier = TeamClassifier(device=device)

    # Collect crops for team classification
    for frame in tqdm(frame_generator, desc='Collecting crops for team classification'):
        result = player_detection_model(frame, imgsz=1280, verbose=False, conf=player_detection_conf)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops = get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    
    team_classifier.fit(crops)

    # Save the team classifier state
    with open(state_save_path, 'wb') as f:
        pickle.dump(team_classifier, f)
    print(f"Team classification state saved to {state_save_path}")

def run_localization_with_saved_state(
    source_video_path: str,
    device: str,
    player_detection_conf: float = 0.5,
    ball_detection_conf: float = 0.5,
    keypoints_detection_conf: float = 0.5,
    lines_detection_conf: float = 0.5,
    show_field_mode: str = 'both',
    state_save_path: str = "team_classification_state.pkl"
) -> Iterator[np.ndarray]:
    """
    Run localization on a video using saved team classification state.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        state_save_path (str): Path to load the team classifier state.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames with player localization.
    """

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    keypoints_detection_model = load_model_kp(KEYPOINT_CONFIG_PATH, KEYPOINT_DETECTION_MODEL_PATH)
    lines_detection_model = load_model_line(LINE_CONFIG_PATH, LINE_DETECTION_MODEL_PATH)

    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def ball_detection_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False, conf=ball_detection_conf)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=ball_detection_callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    # Load the saved team classifier state
    with open(state_save_path, 'rb') as f:
        team_classifier = pickle.load(f)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    for frame in frame_generator:
        try:
            h, w, _ = frame.shape

            # Step 1: Field detection and projection
            result_field = inference(frame, keypoints_detection_model, lines_detection_model, keypoints_detection_conf, lines_detection_conf, h, w)
            H = projection_from_cam_params(result_field)
            frame = project(frame, H, show_field_mode=show_field_mode)

            # Step 2: Player detection (on original frame) and team classification
            result_object = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result_object)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops = get_crops(frame, players)  # Use original frame for cropping
            players_team_id = team_classifier.predict(crops)

            # Step 3: Ball detection (on warped frame)
            detections_ball = slicer(frame).with_nms(threshold=0.1)
            detections_ball = ball_tracker.update(detections_ball)

            # Step 4: Goalkeeper and referee detection
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            # Step 5: Merge detections
            ball = detections_ball[detections_ball.class_id == BALL_CLASS_ID]
            
            all_detections = sv.Detections.merge([players, goalkeepers, referees])

            # Step 6: Color lookup for annotations
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees) #+
                #[BALL_CLASS_ID] * len(ball)
            )

            labels = [str(tracker_id) for tracker_id in all_detections.tracker_id]

            # Step 7: Annotate the original frame
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(frame.copy(), all_detections, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, all_detections, labels, custom_color_lookup=color_lookup)
            annotated_frame = ball_annotator.annotate(annotated_frame, detections_ball)

            # Step 8: Inverse projection for visualization
            H_inv = inverse_projection_matrix(H)
            frame_map = visualize_field_and_players(all_detections, color_lookup, H_inv)

        except Exception as e:
            print(f"An error occurred during player localization: {e}")
            continue

        yield annotated_frame, frame_map

def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame

def run_localization(
    source_video_path: str,
    device: str,
    player_detection_conf: float = 0.5,
    ball_detection_conf: float = 0.5,
    keypoints_detection_conf: float = 0.5,
    lines_detection_conf: float = 0.5,
    show_field_mode: str = 'both'
) -> Iterator[np.ndarray]:
    """
    Run localization on a video and yield annotated frames with player localization.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames with player localization.
    """

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    keypoints_detection_model = load_model_kp(KEYPOINT_CONFIG_PATH, KEYPOINT_DETECTION_MODEL_PATH)
    lines_detection_model = load_model_line(LINE_CONFIG_PATH, LINE_DETECTION_MODEL_PATH)

    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def ball_detection_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False, conf=ball_detection_conf)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=ball_detection_callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    # Collect crops for team classification before any projection or warping
    crops = []
    for frame in tqdm(frame_generator, desc='Collecting crops for team classification'):
        result = player_detection_model(frame, imgsz=1280, verbose=False, conf=player_detection_conf)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    player_positions = {}
    frame_idx = 1

    for frame in frame_generator:
        try:
            h, w, _ = frame.shape

            # Step 1: Field detection 
            result_field = inference(frame, keypoints_detection_model, lines_detection_model, keypoints_detection_conf, lines_detection_conf, h, w)
            H = projection_from_cam_params(result_field)
            
            # Step 2: Player detection (on original frame) and team classification
            result_object = player_detection_model(frame, imgsz=1280, verbose=False, conf=player_detection_conf)[0]
            detections = sv.Detections.from_ultralytics(result_object)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops = get_crops(frame, players)  
            players_team_id = team_classifier.predict(crops)

            # Step 3: Ball detection
            detections_ball = slicer(frame).with_nms(threshold=0.1)
            detections_ball = ball_tracker.update(detections_ball)

            # Step 4: Goalkeeper and referee detection
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            # Step 5: Merge detections
            all_detections = sv.Detections.merge([players, goalkeepers, referees]) #, ball])

            # Step 6: Color lookup for annotations
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)# +
                #[BALL_CLASS_ID] * len(ball)
            )
            labels = [str(tracker_id) for tracker_id in all_detections.tracker_id]

            # Step 7: Annotate the original frame
            frame = project(frame, H, show_field_mode=show_field_mode)
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(frame.copy(), all_detections, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, all_detections, labels, custom_color_lookup=color_lookup)
            annotated_frame = ball_annotator.annotate(annotated_frame, detections_ball)

            # Step 8: Inverse projection for visualization
            H_inv = inverse_projection_matrix(H)
            frame_map, position = visualize_field_and_players(all_detections, color_lookup, H_inv, 'output.png')

            player_positions[frame_idx] = position
            #print(position)
            frame_idx += 1

        except Exception as e:
            print(f"An error occurred during player localization: {e}")
            continue

        yield annotated_frame, frame_map

    # Menyimpan data posisi pemain ke file JSON setelah loop selesai
    json_filename = f'{datetime.now().strftime("%Y%m%d")}_player_positions.json'
    try:
        with open(json_filename, 'w') as f:
            json.dump(player_positions, f, indent=4)
        print(f"Player positions saved to {json_filename}")
    except Exception as e:
        print(f"An error occurred while saving player positions: {e}")


def main(
        source_video_path: str, 
        target_video_path: str, 
        device: str, 
        mode: Mode, 
        player_detection_conf: float = 0.5, 
        keypoints_detection_conf: float = 0.5,
        lines_detection_conf: float = 0.5,
        ball_detection_conf: float = 0.5,
        show_field_mode: str = 'both',
        stframe=None,
        stframe_map=None,
        target_video_map_path: str = None,
        stop_event=None
        ) -> None:
    
    # Frame processing
    if mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device, player_detection_conf=player_detection_conf)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device, player_detection_conf=player_detection_conf)
    elif mode == Mode.PITCH_DETECTION:
        frame_generator = run_field_detection(
            source_video_path=source_video_path, device=device, keypoints_detection_conf=keypoints_detection_conf, lines_detection_conf=lines_detection_conf, show_field_mode=show_field_mode)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device, ball_detection_conf=ball_detection_conf)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device, player_detection_conf=player_detection_conf)
    elif mode == Mode.LOCALIZATION:
        frame_generator = run_localization(
            source_video_path=source_video_path, device=device, 
            player_detection_conf=player_detection_conf, 
            ball_detection_conf=ball_detection_conf, 
            keypoints_detection_conf=keypoints_detection_conf, 
            lines_detection_conf=lines_detection_conf, 
            show_field_mode=show_field_mode)
        
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    # Video reconstruction
    if mode == Mode.LOCALIZATION:
        first_frame = next(frame_generator)
        frame, frame_map = first_frame
        frame_map_height, frame_map_width, _ = frame_map.shape

        video_info = sv.VideoInfo.from_video_path(source_video_path)
        video_info_map = sv.VideoInfo(
            width=frame_map_width,
            height=frame_map_height,
            fps=video_info.fps, 
            total_frames=video_info.total_frames,
        )

        total_frames = video_info.total_frames
        progress_bar = st.progress(0, text = 'Detection starting...')
        time_remaining_text = st.empty()
        start_time = time.time()

        with sv.VideoSink(target_video_path, video_info) as sink, sv.VideoSink(target_video_map_path, video_info_map) as sink_map:
            for idx, frames in enumerate(frame_generator):
                if stop_event and stop_event.is_set():  # Check if the stop button was pressed
                    st.toast("Detection Stopped!")
                    break

                print(frame_map_width, frame_map_height)

                progress_percentage = (idx + 1) / total_frames
                progress_bar.progress(progress_percentage, text = f"Detection in progress ({round(progress_percentage*100)}%)")
                elapsed_time = time.time() - start_time
                avg_time_per_frame = elapsed_time / (idx + 1)
                remaining_frames = total_frames - (idx + 1)
                estimated_time_remaining = remaining_frames * avg_time_per_frame / 60
                time_remaining_text.text(f"Estimated time remaining: {estimated_time_remaining:.2f} minutes")
                frame, frame_map = frames
               
                if not stframe and not stframe_map:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                stframe.image(frame, channels="BGR")
                stframe_map.image(frame_map, channels="BGR")
                
                try:
                    sink.write_frame(frame)
                    sink_map.write_frame(frame_map)

                except Exception as e:
                    print(f"Write error: {e}")
                
        if not stframe and not stframe_map:
            cv2.destroyAllWindows()

    else:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        total_frames = video_info.total_frames
        progress_bar = st.progress(0, text = 'Detection starting...')
        time_remaining_text = st.empty()
        start_time = time.time()
        
        with sv.VideoSink(target_video_path, video_info) as sink:
            for idx, frame in enumerate(frame_generator):
                if stop_event and stop_event.is_set():  # Check if the stop button was pressed
                    st.toast("Detection Stopped!")
                    break

                progress_percentage = (idx + 1) / total_frames
                progress_bar.progress(progress_percentage, text = f"Detection in progress ({round(progress_percentage*100)}%)")

                elapsed_time = time.time() - start_time
                avg_time_per_frame = elapsed_time / (idx + 1)
                remaining_frames = total_frames - (idx + 1)
                estimated_time_remaining = remaining_frames * avg_time_per_frame / 60
                time_remaining_text.text(f"Estimated time remaining: {estimated_time_remaining:.2f} minutes")

                if not stframe:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                if stframe:
                    stframe.image(frame, channels="BGR")
                
                try:
                    sink.write_frame(frame)

                except Exception as e:
                    print(f"Write error: {e}")
                    
        if not stframe and not stframe_map:
            cv2.destroyAllWindows()


'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--source_video_path', type=str, required=True)
    #parser.add_argument('--target_video_path', type=str, required=True)
    #parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.LOCALIZATION)
    args = parser.parse_args()
    main(
        source_video_path= 'examples/soccer/data/demo1.mp4', #args.source_video_path,
        target_video_path= 'examples/soccer/data/demo1_output.mp4', #args.target_video_path,
        device= 'cuda:0', #args.device,
        mode=args.mode
    )
'''