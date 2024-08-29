import json
import numpy as np
from scipy.spatial import distance

# Fungsi untuk melakukan interpolasi posisi
def interpolate_positions(positions1, positions2, alpha=0.5):
    interpolated = []
    id_map = match_player_ids(positions1, positions2)
    for p1 in positions1:
        # Cari ID pemain yang cocok di frame berikutnya
        matched_idx = id_map.get(p1['player_idx'])
        # Temukan posisi pemain di frame berikutnya yang sesuai dengan ID
        p2 = next((p for p in positions2 if p['player_idx'] == matched_idx), None)
        if p2:
            interp_x = p1['x'] * (1 - alpha) + p2['x'] * alpha
            interp_y = p1['y'] * (1 - alpha) + p2['y'] * alpha
            interpolated.append({'player_idx': p1['player_idx'], 'team_sides': p1['team_sides'], 'x': interp_x, 'y': interp_y})
    return interpolated

# Fungsi untuk menetapkan ID pemain berdasarkan posisi
def match_player_ids(positions1, positions2, threshold=5.0):
    id_map = {}
    for p1 in positions1:
        closest_dist = float('inf')
        closest_idx = None
        for p2 in positions2:
            dist = distance.euclidean((p1['x'], p1['y']), (p2['x'], p2['y']))
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = p2['player_idx']
        if closest_dist < threshold:
            id_map[p1['player_idx']] = closest_idx
    return id_map

# Fungsi utama untuk membaca, memproses dan menyimpan data
def process_player_positions(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    frames = sorted(data.keys(), key=int)
    smoothed_data = {}

    for i in range(len(frames) - 1):
        frame1 = int(frames[i])
        frame2 = int(frames[i + 1])
        positions1 = data[str(frame1)]
        positions2 = data[str(frame2)]

        # Match player IDs between frames
        id_map = match_player_ids(positions1, positions2)

        # Interpolate positions
        interpolated = interpolate_positions(positions1, positions2)
        smoothed_data[frame1] = positions1
        smoothed_data[frame2] = interpolated

    # Save smoothed data to a new file
    with open(output_file, 'w') as f:
        json.dump(smoothed_data, f, indent=4)

# Contoh penggunaan
input_file = 'player_positions.json'
output_file = 'smoothed_player_positions.json'
process_player_positions(input_file, output_file)
