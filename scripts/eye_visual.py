import json
import os
import re
import matplotlib.pyplot as plt
import settings as st
import numpy as np
import utils as ut
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.stats import gaussian_kde

def eye_position_median(left_eye_pos, right_eye_pos):
    """
    Takes lists of left and right eye positions. Calculates the median position for every instance and returns it in a list.
    :param list left_eye_pos:
    :param list right_eye_pos:
    :return:
    :rtype: list,
    """
    eye_pos_median = []
    for left_pos, right_pos in zip(left_eye_pos, right_eye_pos):
        pos_median = [
            (left_pos[0] + right_pos[0]) / 2,
            (left_pos[1] + right_pos[1]) / 2,
            (left_pos[2] + right_pos[2]) / 2
        ]
        eye_pos_median.append(pos_median)
    return eye_pos_median


def quarternion_medians(quats1, quats2):
    """
    Takes two lists of quaternions. Calculates the median position for every instance and returns it in a list.
    :param list quats1: First list of quaternions.
    :param list quats2: Second list of quaternions.
    :return: List of median positions.
    :rtype: list,
    """
    medians = []
    for q1_array, q2_array in zip(quats1, quats2):
        if q1_array[3] != 0 and q2_array[3] != 0:
            q1 = R.from_quat(q1_array)
            q2 = R.from_quat(q2_array)
            if np.dot(q1.as_quat(), q2.as_quat()) < 0:
                q2 = R.from_quat(-q2.as_quat())
            slerp = Slerp([0, 1], R.concatenate([q1, q2]))
            median = slerp(0.5)
            medians.append(median.as_quat())
    return medians


def fix_eye_on_head(head_positions, head_rotations, eye_offsets):
    """
    Calculates the eye position inside VR environment from tracked eye positions and head positions/rotations.
    :param list head_positions:
    :param list head_rotations:
    :param list eye_offsets:
    :return:
    :rtype: list
    """
    eye_positions_world = []
    for head_pos, quat, eye_offset in zip(head_positions, head_rotations, eye_offsets):
        head_pos_np = np.array(head_pos)
        eye_offset_np = np.array(eye_offset)
        rotation = R.from_quat(quat)
        eye_world = head_pos_np - rotation.apply(eye_offset_np)
        eye_positions_world.append(eye_world.tolist())
    return eye_positions_world


def compute_cyclopean_gaze_directions(left_eye_rots, right_eye_rots, head_rots, local_forward=np.array([0, 0, -1])):
    """

    :param list left_eye_rots:
    :param list right_eye_rots:
    :param list head_rots:
    :param local_forward:
    :return:
    :rtype: list
    """
    rotated_directions = []
    for l_eye, r_eye, head in zip(left_eye_rots, right_eye_rots, head_rots):
        if l_eye[2] == 0.0 and l_eye[3] == 0.0 or r_eye[2] == 0.0 and r_eye[3] == 0.0:
            rotated_direction = None
            rotated_directions.append(rotated_direction)
        else:
            l_eye = R.from_quat(l_eye)
            r_eye = R.from_quat(r_eye)
            head = R.from_quat(head)
            l_world = head * l_eye
            r_world = head * r_eye
            slerp = Slerp([0, 1], R.concatenate([l_world, r_world]))
            cyclopean_rot = slerp([0.5])[0]  # Rotation at halfway point
            gaze_dir = cyclopean_rot.apply(local_forward)
            gaze_dir /= np.linalg.norm(gaze_dir)  # Normalize
            rotated_directions.append(gaze_dir)
    return rotated_directions


def get_directional_vector(eye_rot_median, head_rotation, body_rotation, local_direction=np.array([0, 0, 1])):
    """
    Calculates the directional vector for the eye movement inside VR environment.
    :param list eye_rot_median:
    :param list head_rotation:
    :param list body_rotation:
    :param array local_direction:
    :return:
    :rtype: list
    """
    rotated_directions = []
    for rotation_eye, rotation_head, rotation_body in zip(eye_rot_median, head_rotation, body_rotation):
        local_direction = np.array(local_direction)
        if rotation_eye[2] == 0.0 and rotation_eye[3] == 0.0:
            rotated_direction = None
            rotated_directions.append(rotated_direction)
        else:
            eye = R.from_quat(rotation_eye)
            head = R.from_quat(rotation_head)
            body = R.from_quat(rotation_body)
            rotation_combined = head
            rotated_direction = rotation_combined.apply(local_direction)
            rotated_directions.append(rotated_direction.tolist())
    return rotated_directions


def construct_facing_rectangle(body_position, eye_position, width=0.75, height=2):
    """
    Constructs a vertical rectangle at `body_position` (bottom center),
    facing toward `eye_position` (only in XZ plane).
    :param list body_position:
    :param list eye_position:
    :param float width:
    :param float height:
    :return: the 4 corners: bottom_left, bottom_right, top_left, top_right
    """
    body_pos = np.array(body_position)
    eye_pos = np.array(eye_position)

    # Direction from body to eye (XZ plane only)
    direction = eye_pos - body_pos
    direction[1] = 0  # Project to XZ

    if np.linalg.norm(direction) == 0:
        raise ValueError("Eye position cannot be the same as body position in XZ")

    forward_xz = direction / np.linalg.norm(direction)

    # Right vector (perpendicular in XZ)
    right_xz = np.array([-forward_xz[2], 0, forward_xz[0]])

    half_width = width / 2.0
    up = np.array([0, height, 0])

    # Rectangle corners
    bottom_center = body_pos
    bottom_left = bottom_center - right_xz * half_width
    bottom_right = bottom_center + right_xz * half_width
    top_left = bottom_left + up
    top_right = bottom_right + up

    return [tuple(bottom_left), tuple(bottom_right), tuple(top_right), tuple(top_left)]


def intersect_ray_with_plane(ray_origin, ray_direction, plane_point, plane_normal):
    """
    Calculates the intersection point between view ray and plane.
    :param ray_origin:
    :param ray_direction:
    :param plane_point:
    :param plane_normal:
    :return:
    """
    ray_origin = np.asarray(ray_origin, dtype=float)
    ray_dir = np.asarray(ray_direction, dtype=float)
    ray_dir /= np.linalg.norm(ray_dir)
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        return None, None  # Parallel
    t = np.dot(plane_normal, plane_point - ray_origin) / denom
    if t < 0:
        return None, None  # Behind the ray origin
    intersection = ray_origin + t * ray_dir
    return intersection, t


def point_in_rectangle(point, rect_p0, edge1, edge2):
    """
    Calculates if intersection with plane falls within the rectangle.
    :param point:
    :param rect_p0:
    :param edge1:
    :param edge2:
    :return:
    :rtype: bool
    """
    vec = point - rect_p0
    dot00 = np.dot(edge1, edge1)
    dot01 = np.dot(edge1, edge2)
    dot11 = np.dot(edge2, edge2)
    dot02 = np.dot(edge1, vec)
    dot12 = np.dot(edge2, vec)
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return 0 <= u <= 1 and 0 <= v <= 1


def build_plane_from_rectangle(rectangle):
    """
    Creates the plane the rectangle lies within.
    :param rectangle:
    :return:
    """
    p0 = np.asarray(rectangle[0], dtype=float)
    p1 = np.asarray(rectangle[1], dtype=float)
    p3 = np.asarray(rectangle[3], dtype=float)
    edge1 = p1 - p0  # width
    edge2 = p3 - p0  # height
    normal = np.cross(edge1, edge2)
    normal /= np.linalg.norm(normal)
    return p0, edge1, edge2, normal


def get_avg_pos(body_pos):
    """
    Takes list of positions and calculates average position.
    :param body_pos:
    :return:
    :rtype: list
    """
    median = np.median(body_pos, axis=0)
    distances = np.linalg.norm(body_pos - median, axis=1)
    max_index = np.argmax(distances)
    farthest_point = body_pos[max_index]
    print("Far:", farthest_point)
    print("Median:", median)
    return median


def coord_dicts_to_list(coord_dicts):
    """
    Takes list of coordinate dictionaries and converts them to list.
    :param list coord_dicts:
    :return:
    :rtype: list
    """
    coord_list = []
    for coord_dict in coord_dicts:
        if len(coord_dict) == 3:
            new_list = [coord_dict["x"], coord_dict["y"], coord_dict["z"]]
            coord_list.append(new_list)
        elif len(coord_dict) == 4:
            new_list = [coord_dict["x"], coord_dict["y"], coord_dict["z"], coord_dict["w"]]
            coord_list.append(new_list)
    return coord_list


def compute_rectangle_hits_single(body_positions, head_positions, direction_list):
    """
    Calculates rectangle hits for non-average body and eye positions
    :param body_positions:
    :param head_positions:
    :param direction_list:
    :return:
    """
    points_hits = []
    for body, head, direction in zip(body_positions, head_positions, direction_list):
        rectangle = construct_facing_rectangle(body, head)
        p0, edge1, edge2, normal = build_plane_from_rectangle(rectangle)
        if direction is not None:
            intersection, _ = intersect_ray_with_plane(head, direction, p0, normal)
            if intersection is not None:
                hit = point_in_rectangle(intersection, p0, edge1, edge2)
                points_hits.append((intersection, hit))
    return points_hits


def calculate_single_hits(_left_eye_positions,
                          _right_eye_positions,
                          _left_eye_rotations,
                          _right_eye_rotations,
                          _head_positions,
                          _head_rotations,
                          _body_positions,
                          _body_rotations):
    """
    Calculates the visual hits onto the plane and rectangle for non-average body and eye positions.
    :param _left_eye_positions:
    :param _right_eye_positions:
    :param _left_eye_rotations:
    :param _right_eye_rotations:
    :param _head_positions:
    :param _head_rotations:
    :param _body_positions:
    :param _body_rotations:
    :return:
    """
    median_eye_pos = eye_position_median(_left_eye_positions, _right_eye_positions)
    median_eye_rot = quarternion_medians(_left_eye_rotations, _right_eye_rotations)
    #vr_eye_pos = fix_eye_on_head(_head_positions, _head_rotations, median_eye_pos)
    eye_directions = get_directional_vector(median_eye_rot, _head_rotations, _body_rotations)
    hits = compute_rectangle_hits_single(_body_positions, _head_positions, eye_directions)
    hit_count = 0
    for hit in hits:
        if hit[1]:
            hit_count += 1
    if len(_head_positions) != 0:
        percentage = hit_count / len(_head_positions)
    else:
        percentage = 0
    return len(_head_positions), hit_count, percentage


def compute_enclosing_sphere_radius(eye_origin, rect_corners):
    """
    Calculates the enclosing sphere radius for non-average body.
    :param eye_origin:
    :param rect_corners:
    :return:
    """
    rect_corners = np.array(rect_corners)
    dists = np.linalg.norm(rect_corners - eye_origin, axis=1)
    return np.max(dists) * 1.05  # small

def intersect_ray_with_rectangle(ray_origin, ray_dir, rect_corners):
    """
    rect_corners: numpy array of shape (4, 3) - rectangle corners in order
    ray_origin: numpy array of shape (3,)
    ray_dir: numpy array of shape (3,)
    Returns intersection point or None
    """
    ray_origin = np.array(ray_origin)
    ray_dir = np.array(ray_dir)
    rect_corners = np.array(rect_corners)

    # Get two edge vectors
    p0, p1, p2, p3 = rect_corners
    v0 = p1 - p0
    v1 = p3 - p0

    # Compute normal of the rectangle's plane
    normal = np.cross(v0, v1)
    normal = normal / np.linalg.norm(normal)

    # Check if ray and plane are parallel
    denom = np.dot(normal, ray_dir)
    if abs(denom) < 1e-6:
        return None  # Ray is parallel to plane

    # Compute intersection point with the plane
    t = np.dot(p0 - ray_origin, normal) / denom
    if t < 0:
        return None  # Intersection behind origin

    hit_point = ray_origin + t * ray_dir

    # Now check if the hit point is inside the rectangle (as a parallelogram)
    local_hit = hit_point - p0
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, local_hit)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, local_hit)

    denom_rect = dot00 * dot11 - dot01 * dot01
    if denom_rect == 0:
        return None

    u = (dot11 * dot02 - dot01 * dot12) / denom_rect
    v = (dot00 * dot12 - dot01 * dot02) / denom_rect

    if 0 <= u <= 1 and 0 <= v <= 1:
        return hit_point  # Inside rectangle
    else:
        return None


def compute_sphere_intersections(view_directions, avg_eye_pos, sphere_radius):
    """
    Calculates intersection points for non-average body and eye positions.
    :param view_directions:
    :param avg_eye_pos:
    :param sphere_radius:
    :return:
    """
    intersections = []
    for direction in view_directions:
        if direction is None:
            continue
        view_dir = np.array(direction)
        norm = np.linalg.norm(view_dir)
        if norm == 0:
            continue  # Skip zero-length vectors
        view_dir_normalized = view_dir / norm
        intersection = avg_eye_pos + sphere_radius * view_dir_normalized
        intersections.append(intersection)
    return np.array(intersections)


def compute_rectangle_intersections(eye_origin, view_directions, rectangle):
    """
    Calculates intersection points for non-average body and eye positions.
    :param eye_origin:
    :param view_directions:
    :param rectangle:
    :return:
    """
    hits = []
    for dir_vec in view_directions:
        if dir_vec is None:
            continue
        hit = intersect_ray_with_rectangle(eye_origin, dir_vec, rectangle)
        if hit is not None:
            hits.append(hit)
    return np.array(hits)

def compute_sphere_intersections_dynamic(eye_positions, view_directions, sphere_center, sphere_radius):
    hits = []

    for origin, direction in zip(eye_positions, view_directions):
        if direction is None:
            continue
        else:
            origin = np.array(origin)
            direction = np.array(direction)

            oc = origin - sphere_center
            b = np.dot(direction, oc)
            c = np.dot(oc, oc) - sphere_radius ** 2
            discriminant = b ** 2 - c

            if discriminant < 0:
                continue  # no intersection

            sqrt_d = np.sqrt(discriminant)

            # Two possible intersections
            t1 = -b - sqrt_d
            t2 = -b + sqrt_d

            # Choose the closest valid one in front of the ray origin
            t = min(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else None
            if t is None:
                continue

            hit_point = origin + t * direction
            hits.append(hit_point)
    return np.array(hits)


def compute_rectangle_intersections_dynamic(eye_origins, view_directions, rectangle):
    hits = []
    for eye_pos, dir_vec in zip(eye_origins, view_directions):
        if dir_vec is None:
            continue
        else:
            hit = intersect_ray_with_rectangle(eye_pos, dir_vec, rectangle)
            if hit is not None:
                hits.append(hit)
    return np.array(hits)

def plot_interactive_gaze_scene(eye_origin, sphere_hits, rect_corners, rect_hits, view_directions, sphere_radius, player_id):
    """
    Plots a 3D modell of Interactive Gaze Scene.
    :param eye_origin:
    :param sphere_hits:
    :param rect_corners:
    :param rect_hits:
    :param view_directions:
    :param sphere_radius:
    :param player_id:
    :return:
    """
    # Convert to spherical angles
    x, y, z = sphere_hits.T
    r = np.linalg.norm(sphere_hits, axis=1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)

    # Smooth heatmap via KDE
    kde = gaussian_kde(np.vstack([theta, phi]))
    heat_values = kde(np.vstack([theta, phi]))
    norm_heat = (heat_values - heat_values.min()) / (heat_values.max() - heat_values.min())

    # Start Plotly figure
    fig = go.Figure()

    # Eye origin
    fig.add_trace(go.Scatter3d(
        x=[eye_origin[0]], y=[eye_origin[1]], z=[eye_origin[2]],
        mode='markers',
        marker=dict(size=6, color='black'),
        name='Eye Origin'
    ))

    # Sphere hits with KDE-based heatmap
    fig.add_trace(go.Scatter3d(
        x=sphere_hits[:, 0], y=sphere_hits[:, 1], z=sphere_hits[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=norm_heat,
            colorscale='Jet',
            colorbar=dict(title='Gaze Density'),
            opacity=0.8
        ),
        name='Sphere Hits'
    ))

    # Rectangle outline
    rect = np.vstack([rect_corners, rect_corners[0]])  # close the loop
    fig.add_trace(go.Scatter3d(
        x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
        mode='lines',
        line=dict(color='green', width=5),
        name='Rectangle'
    ))

    # Rectangle hits
    if rect_hits.size > 0:
        fig.add_trace(go.Scatter3d(
            x=rect_hits[:, 0], y=rect_hits[:, 1], z=rect_hits[:, 2],
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Rectangle Intersections'
        ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='Interactive Gaze Heatmap & Object Hits (KDE)',
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99,  # Move to top-left corner
            bgcolor='rgba(255,255,255,0.7)',  # Semi-transparent background
            bordercolor='black',
            borderwidth=1,
            font=dict(size=30)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    filtered_view = []
    for view in view_directions:
        if view is None:
            continue
        filtered_view.append(view)
    views = np.array(filtered_view)

    # Compute average gaze direction (unit vector)
    avg_gaze_dir = np.mean(views, axis=0)
    avg_gaze_dir /= np.linalg.norm(avg_gaze_dir)

    # Arrow from sphere center to hit point on sphere
    arrow_start = eye_origin
    arrow_end = eye_origin + avg_gaze_dir * sphere_radius

    # Plot the average gaze direction vector
    fig.add_trace(go.Scatter3d(
        x=[arrow_start[0], arrow_end[0]],
        y=[arrow_start[1], arrow_end[1]],
        z=[arrow_start[2], arrow_end[2]],
        mode='lines',
        line=dict(color='black', width=6, dash='dash'),
        name='Average Gaze Direction'
    ))
    fig.show()


def simulate_gaze_scene_3d(
    left_eye_positions,
    right_eye_positions,
    left_eye_rotations,
    right_eye_rotations,
    head_positions,
    head_rotations,
    body_positions,
    body_rotations,
    body_rotations_2,
    player_id
):
    """
    Helper function for simulating the gaze scene 3d.
    :param left_eye_positions:
    :param right_eye_positions:
    :param left_eye_rotations:
    :param right_eye_rotations:
    :param head_positions:
    :param head_rotations:
    :param body_positions:
    :param body_rotations:
    :param body_rotations_2:
    :param player_id:
    :return:
    """
    median_eye_pos = eye_position_median(left_eye_positions, right_eye_positions)
    median_eye_rot = quarternion_medians(left_eye_rotations, right_eye_rotations)
    avg_head_pos = get_avg_pos(head_positions)
    eye_directions = get_directional_vector(median_eye_rot, head_rotations, body_rotations_2)
    #eye_directions = compute_cyclopean_gaze_directions(left_eye_rotations, right_eye_rotations, head_rotations)
    avg_body_pos = get_avg_pos(body_positions)
    #avg_body_rot = get_avg_pos(body_rotations)
    rectangle = construct_facing_rectangle(avg_body_pos, avg_head_pos)
    rectangle = np.array(rectangle)
    radius = compute_enclosing_sphere_radius(avg_head_pos, rectangle)
    #sphere_hits = compute_sphere_intersections_dynamic(head_positions, eye_directions, avg_head_pos, radius)
    sphere_hits = compute_sphere_intersections(eye_directions, avg_head_pos, radius) #TEST
    #rect_hits = compute_rectangle_intersections_dynamic(head_positions, eye_directions, rectangle)
    rect_hits = compute_rectangle_intersections(avg_head_pos, eye_directions, rectangle)
    plot_interactive_gaze_scene(avg_head_pos, sphere_hits, rectangle, rect_hits, eye_directions, radius, player_id)


def load_data_files(loopvar1, loopvar2, body1, body2, eye1, eye2, head1, head2):
    """
    Loads all necessary data files for gaze calculations.
    :param loopvar1:
    :param loopvar2:
    :param body1:
    :param body2:
    :param eye1:
    :param eye2:
    :param head1:
    :param head2:
    :return:
    """
    _loopvar1 = ut.load_loopvarids(loopvar1)
    _loopvar2 = ut.load_loopvarids(loopvar2)
    _body1 = ut.load_tracking_data(body1)
    _body2 = ut.load_tracking_data(body2)
    _eye1 = ut.load_tracking_data(eye1)
    _eye2 = ut.load_tracking_data(eye2)
    _head1 = ut.load_tracking_data(head1)
    _head2 = ut.load_tracking_data(head2)
    return _loopvar1, _loopvar2, _body1, _body2, _eye1, _eye2, _head1, _head2


def get_data_from_timerange(_loopvar1, _loopvar2, _body1, _body2, _eye1, _eye2, _head1, _head2, start, end):
    """
    Gets data from timerange.
    :param _loopvar1:
    :param _loopvar2:
    :param _body1:
    :param _body2:
    :param _eye1:
    :param _eye2:
    :param _head1:
    :param _head2:
    :param start:
    :param end:
    :return:
    """
    _body_time_seg1 = ut.get_tracking_data_for_timesegment(start, end, _loopvar1, _body1, "body")
    _body_time_seg2 = ut.get_tracking_data_for_timesegment(start, end, _loopvar2, _body2, "body")
    _eye_time_seg1 = ut.get_tracking_data_for_timesegment(start, end, _loopvar1, _eye1, "eye")
    _eye_time_seg2 = ut.get_tracking_data_for_timesegment(start, end, _loopvar2, _eye2, "eye")
    _head_time_seg1 = ut.get_tracking_data_for_timesegment(start, end, _loopvar1, _head1, "head")
    _head_time_seg2 = ut.get_tracking_data_for_timesegment(start, end, _loopvar2, _head2, "head")
    _body_positions_1 = coord_dicts_to_list(_body_time_seg1["botPos"])
    _body_positions_2 = coord_dicts_to_list(_body_time_seg2["botPos"])
    _body_rotations_1 = coord_dicts_to_list(_body_time_seg1["botRot"])
    _body_rotations_2 = coord_dicts_to_list(_body_time_seg2["botRot"])
    _left_eye_pos_1 = coord_dicts_to_list(_eye_time_seg1["leftEye"]["position"])
    _left_eye_rot_1 = coord_dicts_to_list(_eye_time_seg1["leftEye"]["rotation"])
    _right_eye_pos_1 = coord_dicts_to_list(_eye_time_seg1["rightEye"]["position"])
    _right_eye_rot_1 = coord_dicts_to_list(_eye_time_seg1["rightEye"]["rotation"])
    _left_eye_pos_2 = coord_dicts_to_list(_eye_time_seg2["leftEye"]["position"])
    _left_eye_rot_2 = coord_dicts_to_list(_eye_time_seg2["leftEye"]["rotation"])
    _right_eye_pos_2 = coord_dicts_to_list(_eye_time_seg2["rightEye"]["position"])
    _right_eye_rot_2 = coord_dicts_to_list(_eye_time_seg2["rightEye"]["rotation"])
    _head_positions_1 = coord_dicts_to_list(_head_time_seg1["botPos"])
    _head_positions_2 = coord_dicts_to_list(_head_time_seg2["botPos"])
    _head_rotations_1 = coord_dicts_to_list(_head_time_seg1["botRot"])
    _head_rotations_2 = coord_dicts_to_list(_head_time_seg2["botRot"])
    return (_body_positions_1, _body_positions_2, _body_rotations_1, _body_rotations_2, _left_eye_pos_1, _left_eye_rot_1, _right_eye_pos_1, _right_eye_rot_1,
            _left_eye_pos_2, _left_eye_rot_2, _right_eye_pos_2, _right_eye_rot_2, _head_positions_1, _head_positions_2,
            _head_rotations_1, _head_rotations_2)


def get_eye_data_for_dialogue():
    """
    Retrieves all data for the timerange of actual dialogue.
    :return:
    """
    if os.path.isfile(st.player_json):
        pass
    else:
        ut.excel_to_dict(export=True)
    with open(st.player_json, 'r') as file:
        data = json.load(file)
    total_tracking_points = []
    total_hit_points = []
    all_percentages = []
    for entry in data:
        player1_id = entry['Person 1']
        player2_id = entry['Person 2']
        if len(player1_id) > 1 or len(player2_id) > 1:
            print("Can't handle player's with multiple IDs. Evaluation failed for player's:" + player1_id + player2_id)
            continue
        start = entry['Start']
        end = entry['Ende']
        for folder in os.listdir("data/audio_from_video"):
            if player1_id in folder:
                loopvarid1 = "data/audio_from_video/" + folder + "/loopVarIDs.txt"
            elif player2_id in folder:
                loopvarid2 = "data/audio_from_video/" + folder + "/loopVarIDs.txt"
        for folder in os.listdir("data/json_data"):
            if player1_id in folder:
                body_data_1 = "data/json_data/" + folder + "/body.json"
                eye_data_1 = "data/json_data/" + folder + "/eye.json"
                head_data_1 = "data/json_data/" + folder + "/head.json"
            elif player2_id in folder:
                body_data_2 = "data/json_data/" + folder + "/body.json"
                eye_data_2 = "data/json_data/" + folder + "/eye.json"
                head_data_2 = "data/json_data/" + folder + "/head.json"
        try:
            _loopvar1, _loopvar2, _body1, _body2, _eye1, _eye2, _head1, _head2 = load_data_files(loopvarid1, loopvarid2, body_data_1, body_data_2, eye_data_1, eye_data_2, head_data_1, head_data_2)
            (_body_positions_1, _body_positions_2, _body_rotations_1, _body_rotations_2, _left_eye_pos_1, _left_eye_rot_1, _right_eye_pos_1, _right_eye_rot_1,
             _left_eye_pos_2, _left_eye_rot_2, _right_eye_pos_2, _right_eye_rot_2, _head_positions_1, _head_positions_2,
             _head_rotations_1, _head_rotations_2) = get_data_from_timerange(_loopvar1, _loopvar2, _body1, _body2,
                                                                             _eye1, _eye2, _head1, _head2, start, end)
            total1, hits1, perc1 = calculate_single_hits(_left_eye_pos_1, _right_eye_pos_1, _left_eye_rot_1,
                                                         _right_eye_rot_1, _head_positions_1, _head_rotations_1, _body_positions_2, _body_rotations_1)
            total_tracking_points.append(total1)
            total_hit_points.append(hits1)
            all_percentages.append(perc1)
            total2, hits2, perc2 = calculate_single_hits(_left_eye_pos_2, _right_eye_pos_2, _left_eye_rot_2,
                                                         _right_eye_rot_2, _head_positions_2, _head_rotations_2, _body_positions_1, _body_rotations_2)
            total_tracking_points.append(total2)
            total_hit_points.append(hits2)
            all_percentages.append(perc2)
            simulate_gaze_scene_3d(_left_eye_pos_1, _right_eye_pos_1, _left_eye_rot_1, _right_eye_rot_1,
                                   _head_positions_1, _head_rotations_1, _body_positions_2, player1_id)
            simulate_gaze_scene_3d(_left_eye_pos_2, _right_eye_pos_2, _left_eye_rot_2, _right_eye_rot_2,
                                   _head_positions_2, _head_rotations_2, _body_positions_1, player2_id)
        except UnboundLocalError as e:
            print("Missing data for player " + player1_id + " or " + player2_id, e)
    return total_tracking_points, total_hit_points, all_percentages


def time_to_seconds(timestamp):
    """
    Converts MM:SS.mmm to seconds (float with millisecond precision).
    :param timestamp:
    :return:
    """
    minutes, rest = timestamp.split(':')
    seconds = float(rest)
    return int(minutes) * 60 + seconds


def extract_player_sentence_time_in_seconds(file_path, player_name):
    """
    Extracts the times for all sentences by a player in seconds.
    :param file_path:
    :param player_name:
    :return:
    """
    time_ranges = []
    ending = []
    pattern = re.compile(
        rf"^{re.escape(player_name)}:\s*\[(\d{{2}}:\d{{2}}\.\d{{3}})\s*-->\s*(\d{{2}}:\d{{2}}\.\d{{3}})\]"
    )
    full_file_path = "data/dialogues/" + file_path
    with open(full_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                start_str, end_str = match.groups()
                start_sec = time_to_seconds(start_str)
                end_sec = time_to_seconds(end_str)
                time_ranges.append((start_sec, end_sec))
            line = line.strip()
            if line.endswith('.'):
                ending.append("assertion")
            else:
                ending.append("question")

    return list(zip(time_ranges, ending))


def get_time_slots(time_range):
    """
    Determines time ranges for an input time range.
    :param time_range:
    :return:
    """
    diff = time_range[1] - time_range[0]
    slot_0 = (time_range[0] - 0.5, time_range[0] + 0.5)
    slot_1 = (time_range[0] + 0.5, time_range[1] - 0.5)
    slot_2 = (time_range[1] - 0.5, time_range[1] + 0.5)
    return [slot_0, slot_1, slot_2]


def create_file_path(file_path):
    """
    Checks if file path exists.
    :param file_path:
    :return:
    """
    if os.path.isfile(file_path):
        return file_path


def find_data_files(player_id):
    """
    Finds all data files for a player and if they exist.
    :param player_id:
    :return:
    """
    loopvarid = False
    body_data = False
    eye_data = False
    head_data = False
    for folder in os.listdir("data/audio_from_video"):
        if player_id in folder:
            loopvarid = create_file_path("data/audio_from_video/" + folder + "/loopVarIDs.txt")
    for folder in os.listdir("data/json_data"):
        if player_id in folder:
            body_data = create_file_path("data/json_data/" + folder + "/body.json")
            eye_data = create_file_path("data/json_data/" + folder + "/eye.json")
            head_data = create_file_path("data/json_data/" + folder + "/head.json")
    return loopvarid, body_data, eye_data, head_data


def missing_data_message(loopvarid1, loopvarid2, body_data_1, body_data_2, eye_data_1, eye_data_2, head_data_1, head_data_2, dialogue_data, player1_id, player2_id):
    """
    Print function for missing data types.
    :param loopvarid1:
    :param loopvarid2:
    :param body_data_1:
    :param body_data_2:
    :param eye_data_1:
    :param eye_data_2:
    :param head_data_1:
    :param head_data_2:
    :param dialogue_data:
    :param player1_id:
    :param player2_id:
    :return:
    """
    if not loopvarid1: print("Missing data for player " + player1_id + ": LoopvarID")
    if not loopvarid2: print("Missing data for player " + player2_id + ": LoopvarID")
    if not body_data_1: print("Missing data for player " + player1_id + ": body.json")
    if not body_data_2: print("Missing data for player " + player2_id + ": body.json")
    if not eye_data_1: print("Missing data for player " + player1_id + ": eye.json")
    if not eye_data_2: print("Missing data for player " + player2_id + ": eye.json")
    if not head_data_1: print("Missing data for player " + player1_id + ": head.json")
    if not head_data_2: print("Missing data for player " + player2_id + ": head.json")
    if not dialogue_data: print("Missing data for players " + player1_id + " and " + player2_id + ": Dialogue")
    return


def get_eye_data_for_sentence():
    """
    Calculates eye data for sentences.
    :return:
    """
    if os.path.isfile(st.player_json):
        pass
    else:
        ut.excel_to_dict(export=True)
    with open(st.player_json, 'r') as file:
        data = json.load(file)
    speach_start = []
    speach_mid = []
    speach_end = []
    assertions = []
    questions = []
    data_dict = {}
    test = 0
    for entry in data:
        dialogue_data = False
        player1_id_list = data[entry]['Person 1']
        player2_id_list = data[entry]['Person 2']
        if len(player1_id_list) > 1 or len(player2_id_list) > 1:
            print("Can't handle player's with multiple IDs. Evaluation failed for player's: ", player1_id_list, player2_id_list)
            continue
        player1_id = player1_id_list[0]
        player2_id = player2_id_list[0]
        print("Working on " + player1_id + " and " + player2_id)
        loopvarid1, body_data_1, eye_data_1, head_data_1 = find_data_files(player1_id)
        loopvarid2, body_data_2, eye_data_2, head_data_2 = find_data_files(player2_id)
        for file in os.listdir("data/dialogues"):
            if player1_id in file:
                dialogue_data = file
        if not (loopvarid1 and loopvarid2 and body_data_1 and body_data_2 and eye_data_1 and eye_data_2 and head_data_1 and head_data_2 and dialogue_data):
            missing_data_message(loopvarid1, loopvarid2, body_data_1, body_data_2, eye_data_1, eye_data_2, head_data_1, head_data_2, dialogue_data, player1_id, player2_id)
            continue
        try:
            if type(loopvarid1) and type(loopvarid2) and type(body_data_1) and type(body_data_2) and type(eye_data_1) and type(eye_data_2) and type(head_data_1) and type(head_data_2) is str:
                _loopvar1, _loopvar2, _body1, _body2, _eye1, _eye2, _head1, _head2 = load_data_files(loopvarid1, loopvarid2, body_data_1, body_data_2, eye_data_1, eye_data_2, head_data_1, head_data_2)
                times_player_1 = extract_player_sentence_time_in_seconds(dialogue_data, "Player 1")
                times_player_2 = extract_player_sentence_time_in_seconds(dialogue_data, "Player 2")
                player_1_data, player_1_assertion, player_1_question = [], [], []
                player_2_data, player_2_assertion, player_2_question = [], [], []
                player_1_assertion, player_1_question = [], []
                for (time_range, label) in times_player_1:
                    time_slots = get_time_slots(time_range)
                    slot_stats = []
                    for slot in time_slots:
                        try:
                            (_body_positions_1, _body_positions_2, _body_rotations_1, _body_rotations_2, _left_eye_pos_1, _left_eye_rot_1, _right_eye_pos_1, _right_eye_rot_1,
                             _left_eye_pos_2, _left_eye_rot_2, _right_eye_pos_2, _right_eye_rot_2, _head_positions_1, _head_positions_2,
                             _head_rotations_1, _head_rotations_2) = get_data_from_timerange(_loopvar1, _loopvar2, _body1, _body2,
                                                                                             _eye1, _eye2, _head1, _head2, slot[0], slot[1])
                            total1, hits1, perc1 = calculate_single_hits(_left_eye_pos_1, _right_eye_pos_1, _left_eye_rot_1,
                                                                         _right_eye_rot_1, _head_positions_1, _head_rotations_1,
                                                                         _body_positions_2, _body_rotations_1)
                            slot_stats.append((total1, hits1, perc1))
                        except IndexError:
                            continue
                    if len(slot_stats) > 0: speach_start.append(slot_stats[0])
                    if len(slot_stats) > 1: speach_mid.append(slot_stats[1])
                    if len(slot_stats) > 2: speach_end.append(slot_stats[2])
                    if label == "assertion":
                        assertions.extend(slot_stats)
                        player_1_assertion.extend(slot_stats)
                    else:
                        questions.extend(slot_stats)
                        player_1_question.extend(slot_stats)
                    player_1_data.append(slot_stats)
                for (time_range, label) in times_player_2:
                    time_slots = get_time_slots(time_range)
                    slot_stats = []
                    for slot in time_slots:
                        try:
                            (_body_positions_1, _body_positions_2, _body_rotations_1, _body_rotations_2, _left_eye_pos_1, _left_eye_rot_1, _right_eye_pos_1, _right_eye_rot_1,
                             _left_eye_pos_2, _left_eye_rot_2, _right_eye_pos_2, _right_eye_rot_2, _head_positions_1, _head_positions_2,
                             _head_rotations_1, _head_rotations_2) = get_data_from_timerange(_loopvar1, _loopvar2, _body1, _body2,
                                                                                             _eye1, _eye2, _head1, _head2, slot[0], slot[1])
                            total2, hits2, perc2 = calculate_single_hits(_left_eye_pos_2, _right_eye_pos_2, _left_eye_rot_2,
                                                                         _right_eye_rot_2, _head_positions_2, _head_rotations_2,
                                                                         _body_positions_1, _body_rotations_2)
                            slot_stats.append((total2, hits2, perc2))
                        except IndexError:
                            continue
                    if len(slot_stats) > 0: speach_start.append(slot_stats[0])
                    if len(slot_stats) > 1: speach_mid.append(slot_stats[1])
                    if len(slot_stats) > 2: speach_end.append(slot_stats[2])
                    if label == "assertion":
                        assertions.extend(slot_stats)
                        player_2_assertion.extend(slot_stats)
                    else:
                        questions.extend(slot_stats)
                        player_2_question.extend(slot_stats)
                    player_2_data.append(slot_stats)
                p1_dict = {player1_id: {"all": player_1_data, "assertion": player_1_assertion, "question": player_1_question}}
                p2_dict = {player2_id: {"all": player_2_data, "assertion": player_2_assertion, "question": player_2_question}}
                data_dict.update(p1_dict)
                data_dict.update(p2_dict)
                test += 1
        except UnboundLocalError as e:
            print("Missing data for player " + player1_id + " or " + player2_id + "\nReason: ", e)
    print(test)
    return speach_start, speach_mid, speach_end, assertions, questions, data_dict


def get_stutters(word_json):
    """
    Retrieves timestamps for stutter indications from the json.word file.
    :param str word_json: Path to the word json file
    :return: List of stutter timestamps
    :rtype: list
    """
    with open(st.word_json + "/" + word_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
    times = []
    for word in data['words']:
        if re.match(r'^\[.*\]$', word["text"]):
            times.append(word["timestamp"])
    return times


def get_eye_data_stutter():
    """
    Calculates eye data for stutter indications.
    :return:
    """
    with open(st.player_json, 'r') as file:
        data = json.load(file)
    stutter_stats = []
    stutter_dict = {}
    for entry in data:
        word_data_1 = False
        word_data_2 = False
        player1_id_list = data[entry]['Person 1']
        player2_id_list = data[entry]['Person 2']
        if len(player1_id_list) > 1 or len(player2_id_list) > 1:
            print("Can't handle player's with multiple IDs. Evaluation failed for player's: ", player1_id_list, player2_id_list)
            continue
        player1_id = player1_id_list[0]
        player2_id = player2_id_list[0]
        print("Working on " + player1_id + " and " + player2_id)
        loopvarid1, body_data_1, eye_data_1, head_data_1 = find_data_files(player1_id)
        loopvarid2, body_data_2, eye_data_2, head_data_2 = find_data_files(player2_id)
        for file in os.listdir(st.word_json):
            if player1_id in file:
                word_data_1 = file
            elif player2_id in file:
                word_data_2 = file
        if not (loopvarid1 and loopvarid2 and body_data_1 and body_data_2 and eye_data_1 and eye_data_2 and head_data_1 and head_data_2 and word_data_1 and word_data_2):
            continue
        try:
            _loopvar1, _loopvar2, _body1, _body2, _eye1, _eye2, _head1, _head2 = load_data_files(loopvarid1, loopvarid2, body_data_1, body_data_2, eye_data_1, eye_data_2, head_data_1, head_data_2)
            times_player_1 = get_stutters(word_data_1)
            times_player_2 = get_stutters(word_data_2)
            player1_data = []
            player2_data = []
            for time_range in times_player_1:
                try:
                    (_body_positions_1, _body_positions_2, _body_rotations_1, _body_rotations_2, _left_eye_pos_1,
                    _left_eye_rot_1, _right_eye_pos_1, _right_eye_rot_1,
                    _left_eye_pos_2, _left_eye_rot_2, _right_eye_pos_2, _right_eye_rot_2, _head_positions_1,
                    _head_positions_2,
                    _head_rotations_1, _head_rotations_2) = get_data_from_timerange(_loopvar1, _loopvar2, _body1,
                                                                                         _body2,
                                                                                         _eye1, _eye2, _head1, _head2,
                                                                                         time_range[0] - 0.5, time_range[1] + 0.5)
                    total1, hits1, perc1 = calculate_single_hits(_left_eye_pos_1, _right_eye_pos_1, _left_eye_rot_1,
                                                                     _right_eye_rot_1, _head_positions_1,
                                                                     _head_rotations_1,
                                                                     _body_positions_2, _body_rotations_1)
                    stutter_stats.append((total1, hits1, perc1))
                    player1_data.append((total1, hits1, perc1))
                except IndexError:
                    continue
            for time_range in times_player_2:
                try:
                    (_body_positions_1, _body_positions_2, _body_rotations_1, _body_rotations_2, _left_eye_pos_1,
                    _left_eye_rot_1, _right_eye_pos_1, _right_eye_rot_1,
                    _left_eye_pos_2, _left_eye_rot_2, _right_eye_pos_2, _right_eye_rot_2, _head_positions_1,
                    _head_positions_2,
                    _head_rotations_1, _head_rotations_2) = get_data_from_timerange(_loopvar1, _loopvar2, _body1,
                                                                                         _body2,
                                                                                         _eye1, _eye2, _head1, _head2,
                                                                                         time_range[0], time_range[1])
                    total2, hits2, perc2 = calculate_single_hits(_left_eye_pos_2, _right_eye_pos_2, _left_eye_rot_2,
                                                                     _right_eye_rot_2, _head_positions_2,
                                                                     _head_rotations_2,
                                                                     _body_positions_1, _body_rotations_2)
                    stutter_stats.append((total2, hits2, perc2))
                    player2_data.append((total2, hits2, perc2))
                except IndexError:
                    continue
            p1_dict = {player1_id: player1_data}
            p2_dict = {player2_id: player2_data}
            stutter_dict.update(p1_dict)
            stutter_dict.update(p2_dict)
        except UnboundLocalError as e:
            print(f"Missing data for player {player1_id} or {player2_id} \nReason: ", e)
    return  stutter_stats, stutter_dict


def check_for_zero(hits, total):
    """
    Prevents division by 0.
    :param hits:
    :param total:
    :return:
    """
    if total == 0:
        return 0
    else:
        return hits / total


def total_hits_helper(data):
    """
    Counts total and hits from input data.
    :param data:
    :return:
    """
    total, hits = 0, 0
    for entry in data:
        total += entry[0]
        hits += entry[1]
    return total, hits

def process_data_dicts(data_dict_1, data_dict_2):
    """
    Precesses the data dictionaries created by eye data analyses.
    :param data_dict_1:
    :param data_dict_2:
    :return:
    """
    stat_dict = {}
    percentages = []
    for entry in data_dict_1:
        print(f"Sampling data for {entry}")
        player = data_dict_1[entry]
        sentences = player["all"]
        assertions = player["assertion"]
        questions = player["question"]
        stutters = data_dict_2[entry]
        starts, mids, ends = [], [], []
        for sentence in sentences:
            if len(sentence) > 0: starts.append(sentence[0])
            if len(sentence) > 1: mids.append(sentence[1])
            if len(sentence) > 2: ends.append(sentence[2])
        start_total, start_hits = total_hits_helper(starts)
        mid_total, mid_hits = total_hits_helper(mids)
        end_total, end_hits = total_hits_helper(ends)
        assertion_total, assertion_hits = total_hits_helper(assertions)
        question_total, question_hits = total_hits_helper(questions)
        stutter_total, stutter_hits = total_hits_helper(stutters)
        start_percent = check_for_zero(start_hits, start_total)
        mid_percent = check_for_zero(mid_hits, mid_total)
        end_percent = check_for_zero(end_hits, end_total)
        assertions_percent = check_for_zero(assertion_hits, assertion_total)
        questions_percent = check_for_zero(question_hits, question_total)
        stutter_percent = check_for_zero(stutter_hits, stutter_total)
        percentages.append((start_percent, mid_percent, end_percent, assertions_percent, questions_percent, stutter_percent))
        player_stat = {entry: {"start": {"total": start_total, "hits": start_hits, "percent": start_percent},
                               "mid": {"total": mid_total, "hits": mid_hits, "percent": mid_percent},
                               "end": {"total": end_total, "hits": end_hits, "percent": end_percent},
                               "assertion": {"total": assertion_total, "hits": assertion_hits, "percent": assertions_percent},
                               "question": {"total": question_total, "hits": question_hits, "percentage": questions_percent},
                               "stutter": {"total": stutter_total, "hits": stutter_hits, "percent": stutter_percent}}}
        stat_dict.update(player_stat)
    start_average, mid_average, end_average, assertion_average, question_average, stutter_average = 0, 0, 0, 0, 0, 0
    for entry in percentages:
        start_average += entry[0]
        mid_average += entry[1]
        end_average += entry[2]
        assertion_average += entry[3]
        question_average += entry[4]
        stutter_average += entry[5]
    average_percent = {"start": start_average/len(percentages)*100, "mid": mid_average/len(percentages)*100,
                       "end": end_average/len(percentages)*100, "assertion": assertion_average/len(percentages)*100,
                       "question": question_average/len(percentages)*100, "stutter": stutter_average/len(percentages)*100}
    return stat_dict, average_percent


def plot_eye_data(data, labels, average_percent, save_file):
    """
    Creates a plot ot visualize eye data.
    :param data:
    :param labels:
    :param average_percent:
    :param save_file:
    :return:
    """
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 20})
    bars = plt.bar(labels, data)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 3, f'{height:.2f}%', ha='center', va='bottom')
    for label, average, color in zip(labels, average_percent, colors):
        plt.axhline(average, color=color, linestyle='dashed', linewidth=1, label=f'Average {label}')
    plt.ylim(0, 100)
    plt.ylabel('Percentage')
    plt.title('Percentage Distribution Across Categories')
    #plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)


def data_process():
    """
    Main module for eye data analysis.
    :return:
    """
    print("Dialoge Analyses")
    speach_start, speach_mid, speach_end, assertions, questions, data_dict = get_eye_data_for_sentence()
    print("Stutter Analyses")
    stutter_stats, stutter_dict = get_eye_data_stutter()
    stat_dict, average_percent = process_data_dicts(data_dict, stutter_dict)
    speach_start_total, speach_start_hits = total_hits_helper(speach_start)
    speach_mid_total, speach_mid_hits = total_hits_helper(speach_mid)
    speach_end_total, speach_end_hits = total_hits_helper(speach_end)
    assertion_total, assertion_hits = total_hits_helper(assertions)
    questions_total, questions_hits = total_hits_helper(questions)
    stutter_total, stutter_hits = total_hits_helper(stutter_stats)
    median_speach_start_percent = speach_start_hits / speach_start_total * 100
    median_speach_mid_percent = speach_mid_hits / speach_mid_total * 100
    median_speach_end_percent = speach_end_hits / speach_end_total * 100
    median_assertions_percent = assertion_hits / assertion_total * 100
    median_questions_percent = questions_hits / questions_total * 100
    median_stutter_percent = stutter_hits / stutter_total * 100
    plot_eye_data([median_speach_start_percent, median_speach_mid_percent, median_speach_end_percent],
                  ["Sentence Start", "Sentence Mid", "Sentence End"],
                  [average_percent["start"], average_percent["mid"], average_percent["end"]],
                   "data/eye_data_sentence.png")
    plot_eye_data([median_assertions_percent, median_questions_percent, median_stutter_percent],
                  ["Assertions", "Questions", "Hesitation"],
                  [average_percent["assertion"], average_percent["question"], average_percent["stutter"]],
                  "data/eye_data_aqs.png")


def quick_3d_load(player1_id, player2_id, start, end):
    """
    Quick 3d loader for eye data modell.
    :param player1_id:
    :param player2_id:
    :param start:
    :param end:
    :return:
    """
    loopvarid1, body_data_1, eye_data_1, head_data_1 = find_data_files(player1_id)
    loopvarid2, body_data_2, eye_data_2, head_data_2 = find_data_files(player2_id)
    _loopvar1, _loopvar2, _body1, _body2, _eye1, _eye2, _head1, _head2 = load_data_files(loopvarid1, loopvarid2,
                                                                                         body_data_1, body_data_2,
                                                                                         eye_data_1, eye_data_2,
                                                                                         head_data_1, head_data_2)
    try:
        (_body_positions_1, _body_positions_2, _body_rotations_1, _body_rotations_2, _left_eye_pos_1, _left_eye_rot_1,
         _right_eye_pos_1, _right_eye_rot_1, _left_eye_pos_2, _left_eye_rot_2, _right_eye_pos_2, _right_eye_rot_2,
         _head_positions_1, _head_positions_2, _head_rotations_1, _head_rotations_2) = get_data_from_timerange(_loopvar1,
                                                                        _loopvar2, _body1, _body2,_eye1, _eye2, _head1,
                                                                        _head2, start, end)
        simulate_gaze_scene_3d(_left_eye_pos_1, _right_eye_pos_1, _left_eye_rot_1, _right_eye_rot_1,
                                    _head_positions_1, _head_rotations_1, _body_positions_2, _body_rotations_2, _body_rotations_1,
                                    player1_id)
        simulate_gaze_scene_3d(_left_eye_pos_2, _right_eye_pos_2, _left_eye_rot_2, _right_eye_rot_2,
                                    _head_positions_2, _head_rotations_2, _body_positions_1, _body_rotations_1,
                                    _body_rotations_2,
                                    player2_id)
    except IndexError:
        pass


def load_id_qick_3d_plot():
    """
    Main module for 3d plot creation.
    :return:
    """
    with open(st.player_json, 'r') as file:
        data = json.load(file)
    for entry in data:
        player1_id_list = data[entry]['Person 1']
        player2_id_list = data[entry]['Person 2']
        if "start" in data[entry] and "end" in data[entry]:
            start = data[entry]['start']
            end = data[entry]['end']
        else:
            continue
        if len(player1_id_list) > 1 or len(player2_id_list) > 1:
            print("Can't handle player's with multiple IDs. Evaluation failed for player's: ", player1_id_list, player2_id_list)
            continue
        player1_id = player1_id_list[0]
        player2_id = player2_id_list[0]
        print("Working on " + player1_id + " and " + player2_id)
        quick_3d_load(player1_id, player2_id, start, end)

