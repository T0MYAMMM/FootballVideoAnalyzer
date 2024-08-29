import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import supervision as sv
from io import BytesIO
from PIL import Image
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.patches as pls

def visualize_field_and_players(positions, output_path, width, height):
    keypoint_world_coords_2D = [[x - 52.5, y - 34] for x, y in [
        [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
        [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34], [105., 30.34],
        [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66], [0., 43.16], [5.5, 43.16],
        [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16], [88.5, 54.16], [105., 54.16], [0., 68.],
        [52.5, 68.], [105., 68.], [16.5, 26.68], [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15],
        [88.5, 41.31], [19.99, 32.29], [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46],
        [61.31, 36.46], [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
        [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.], [88.5, 34.], [94., 34.]
    ]]

    line_world_coords_3D = [[[x1 - 52.5, y1 - 34, z1], [x2 - 52.5, y2 - 34, z2]] for [[x1, y1, z1], [x2, y2, z2]] in [
        [[0., 54.16, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [0., 13.84, 0.]],
        [[88.5, 54.16, 0.], [105., 54.16, 0.]], [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], [[88.5, 13.84, 0.], [105., 13.84, 0.]],
        [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]], [[0., 30.34, 0.], [0., 30.34, -2.44]],
        [[105., 37.66, -2.44], [105., 30.34, -2.44]], [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],
        [[52.5, 0., 0.], [52.5, 68, 0.]], [[0., 68., 0.], [105., 68., 0.]], [[0., 0., 0.], [0., 68., 0.]], [[105., 0., 0.], [105., 68., 0.]],
        [[0., 0., 0.], [105., 0., 0.]], [[0., 43.16, 0.], [5.5, 43.16, 0.]], [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
        [[5.5, 24.84, 0.], [0., 24.84, 0.]], [[99.5, 43.16, 0.], [105., 43.16, 0.]], [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
        [[99.5, 24.84, 0.], [105., 24.84, 0.]]
    ]]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))  # Adjust figsize to match frame size
    ax.set_facecolor('lightgreen')

    # Draw the field lines and ellipses (arcs)
    for line in line_world_coords_3D:
        (x1, y1, z1), (x2, y2, z2) = line
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=3)

    # Draw the center circle
    center_circle = plt.Circle((0, 0), 9.15, color="white", fill=False, linewidth=3)
    ax.add_patch(center_circle)

    # Draw the penalty box arcs
    penalty_arc_left = pls.Arc((-52.5 + 11, 0), width=18.3, height=18.3, angle=-90, theta1=37, theta2=143, color='white', linewidth=3)
    penalty_arc_right = pls.Arc((52.5 - 11, 0), width=18.3, height=18.3, angle=-90, theta1=217, theta2=323, color='white', linewidth=3)
    ax.add_patch(penalty_arc_left)
    ax.add_patch(penalty_arc_right)

    COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
    TEAM_COLORS = {'left': '#ff1493', 'right': '#00bfff'}

    # Draw player coordinates
    for pos in positions:
        if pos != []:
            print(pos)
            x, y = pos['x'], pos['y']
            if 'team_sides' in pos:
                team_color = TEAM_COLORS.get(pos['team_sides'], '#ff1493')
            else:
                team_color = '#ff1493'
            ax.scatter(x, -y, color=team_color, zorder=5, s=80)

    # Add legends and key points
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Team Right', markerfacecolor=TEAM_COLORS['right'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Team Left', markerfacecolor=TEAM_COLORS['left'], markersize=10)
    ]

    ax.set_xlim(-52.5, 52.5)
    ax.set_ylim(-34, 34)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X (105 meters) (-52.5...52.5)")
    ax.set_ylabel("Y (meters) (-34...34)")
    ax.set_title("Football Field with Players")
    ax.legend(handles=legend_elements, loc='upper right')

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_np = np.array(Image.open(buf).convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    return img_bgr



def create_video_from_frames(input_json, output_video, frame_rate=30):
    with open(input_json, 'r') as f:
        data = json.load(f)

    frames = sorted(data.keys(), key=int)
    
    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
    #out = cv2.VideoWriter(output_video, fourcc, frame_rate, (1050, 680))  # Adjust the frame size to your needs

    for frame in frames:
        if frame != []:
            positions = data[frame]
            frame_img = visualize_field_and_players(positions, None, 1200, 800)
            break

    #first_frame = next(frames)
    frame_map = frame_img
    frame_map_height, frame_map_width, _ = frame_map.shape

    video_info = sv.VideoInfo.from_video_path('PSK-MU.mp4')
    video_info_map = sv.VideoInfo(
        width=frame_map_width,
        height=frame_map_height,
        fps=video_info.fps, 
        total_frames=video_info.total_frames,
    )

    with sv.VideoSink(output_video, video_info_map) as sink_map:
        for frame in frames:
            positions = data[frame]
            frame_img = visualize_field_and_players(positions, None, frame_map_width, frame_map_height)
            sink_map.write_frame(frame_img)
        #cv2.destroyAllWindows()
    
# Contoh penggunaan
input_json = 'smoothed_player_positions.json'
output_video = 'player_positions_video.mp4'
create_video_from_frames(input_json, output_video)
