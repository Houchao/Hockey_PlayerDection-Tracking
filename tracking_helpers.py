import numpy as np
from dataclasses import dataclass
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

IdPair = namedtuple("IdPair", ['id', 'closest_id'])
lost_with_box_id_pair = namedtuple("Lost_Id_pair_w_box", ['id', 'closest_id'])
lost_without_box_id_pair = namedtuple("Lost_Id_pair_wo_box", ['id', 'frames_lost_for'])


def xyxy_to_xy_centroid(xyxy):
    """Assumes tlhw format"""
    centroid_y = (xyxy[1] + xyxy[-1]) / 2
    centroid_x = (xyxy[0] + xyxy[-2]) / 2
    arr = np.array([centroid_x, centroid_y])
    return arr


def find_closest_box(id, other_ids, thresh, id_cent_dict):
    """Find the closest box to the missing box in the previous frame"""
    distances = [(np.linalg.norm(id_cent_dict[id] - id_cent_dict[o_id]), o_id) for o_id in other_ids]
    closest_dist_idx = min(distances, key=lambda d: d[0])
    if closest_dist_idx[0] < thresh:
        return closest_dist_idx[1]
    else:
        return None


def check_for_new_box_appearances(ids_in_last_frame: set, ids_curr_frame: set, id_centroid_dict: dict,
                                  lost_id_with_nearest_box: set, lost_ids_without_nearest_box: set, bboxes, frame_idx, frame_thresh=5):
    """Attempt at lost_id alg 2"""
    lost_ids = ids_in_last_frame.difference(ids_curr_frame)
    pot_new_ids = ids_curr_frame.difference(ids_in_last_frame)
    new_ids = [id for id in pot_new_ids if id not in id_centroid_dict]
    # Take care of ids that have been lost by tracker/yolov5
    check_frame_numbers(lost_ids_without_nearest_box, frame_thresh, frame_idx)
    # Update centroid info
    for box, id in zip(bboxes, ids_curr_frame):
        id_centroid_dict[id] = xyxy_to_xy_centroid(box)
    # Check to see if lost id has a nearest box
    for id in lost_ids:
        closest_box = find_closest_box(id, [curr_id for curr_id in ids_curr_frame if id != curr_id], 75, id_centroid_dict)
        if closest_box is None:
            lost_ids_without_nearest_box.add(lost_without_box_id_pair(id, 0))
            logging.debug(" At frame index %d, adding ID pair [%d, no_box] to the ids to ignore.", frame_idx, id)
        else:
            lost_id_with_nearest_box.add(lost_with_box_id_pair(id, closest_box))
            logging.debug(" At frame index %d, adding ID pair [%d, %d] to the ids to ignore.", frame_idx, id,
                          closest_box)
    # Check to see if previously lost id has reappeared
    lost_with_box = lost_id_with_nearest_box.copy()
    for id_pair in lost_with_box:
        if id_pair.id in ids_curr_frame:
            lost_id_with_nearest_box.remove(id_pair)
            logging.debug(" At frame index %d, removing ID pair [%d, %d] due to id = %d returning.", frame_idx, id_pair.id,
                          id_pair.closest_id, id_pair.id)
        else:
            closest_box = find_closest_box(id_pair.id, [curr_id for curr_id in ids_curr_frame if id_pair.id != curr_id], 75, id_centroid_dict)
            if closest_box is not None:
                logging.debug(f" At frame index {frame_idx}, Removing ID pair [{id_pair.id}, {id_pair.closest_id}] from the ids to ignore because {closest_box} appeared too close.")
                lost_id_with_nearest_box.remove(id_pair)

    ids_in_last_frame = ids_curr_frame.copy()
    return

    # Keep running list of ids that are lost with or without a nearest box
    # for id in lost_ids:
    #     lost_id_list.add(lost_without_box_id_pair(id, 0))

# def box_within_thresh(lost_ids: set, ids_curr_frame: set, id_centroid_dict, thresh= 125):
#     for id in lost_ids:
#         closest_box_id = find_closest_box(id, [curr_id for curr_id in ids_curr_frame if id != curr_id], 125, id_centroid_dict)
#         if closest_box_id is not None:
#             ids_to_ignore.add(tracking_helpers.Id_pair(id, closest_box_id))
#             logging.debug(" At frame index %d, adding ID pair [%d, %d] to the ids to ignore.", frame_idx, id, closest_box_id)

def check_frame_numbers(lost_id_set: set, frame_thresh: int, frame_idx):
    # Remove ids that have been lost for too long
    new_lost_id_set = lost_id_set.copy()
    for id_tuple in new_lost_id_set:

        if id_tuple.frames_lost_for > frame_thresh:
            # lost_id_list.remove((id, frames_present))
            lost_id_set.remove(lost_without_box_id_pair(id_tuple.id, id_tuple.frames_lost_for))
            logging.debug(" At frame index %d, removing ID pair [%d, no_box] due to not being seen for %d frames.", frame_idx, id_tuple.id,
                          frame_thresh)
        else:
            lost_id_set.remove(lost_without_box_id_pair(id_tuple.id, id_tuple.frames_lost_for))
            lost_id_set.add(lost_without_box_id_pair(id_tuple.id, id_tuple.frames_lost_for + 1))



@dataclass(frozen=True, eq=True)
class Id_pair:
    id: int
    closest_id: int


if __name__ == "__main__":
    xyxy_to_xy_centroid(np.array([803, 516, 873, 653]))
