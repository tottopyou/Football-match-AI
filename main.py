import cv2

from utils import read_video,save_video
from trackers import Tracker
from teams import TeamAssigner
from head_player import HeadPlayer
import numpy as np
def main():

    print("Reading the video")
    video_frames = read_video('Input_video/08fd33_4.mp4')
    print("Tracked the video")
    tracker = Tracker('models/best_second.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    teams = TeamAssigner()
    teams.assign_team_color(video_frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = teams.get_player_team(video_frames[frame_num],track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = teams.team_colors[team]

    player_assigner = HeadPlayer()
    team_ball_control = [1]
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    print("Drawing annotations")
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)
    print("Saving the video")
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == "__main__":
    main()