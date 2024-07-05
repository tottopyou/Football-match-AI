from utils import read_video,save_video
from trackers import Tracker
def main():

    print("Reading the video")
    video_frames = read_video('Input_video/08fd33_4.mp4')
    print("Tracked the video")
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    print("Drawing annotations")
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    print("Saving the video")
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == "__main__":
    main()