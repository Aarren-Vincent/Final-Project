# part6_pipeline.py
import os
import torch
from torch.utils.data import DataLoader
from part2_helpers import read_video_frames, list_videos
from part3_dataset import SimpleVideoList, SnippetDataset
from part4_framehopper import FrameHopper
from part5_mil import MIL_FCNet, train_mil, score_video_model

def demo_train_small(root_videos_folder):
    # 1) Prepare small file list
    s = SimpleVideoList(root_videos_folder)
    files = s.train_list()  # list of (path,label)
    if len(files) < 2:
        print("Please add at least one normal and one anomaly video in videos/normal and videos/anomaly")
        return

    # 2) Quick FrameHopper training on first video (toy demo)
    example_video = files[0][0]
    frames = read_video_frames(example_video, resize=(320,240), max_frames=300)
    fh = FrameHopper(n_states=8, max_skip=4, theta=0.12)
    fh.fit_state_clusters([frames])
    fh.train(frames, epochs=2)
    fh.save("framehopper_q.pkl")
    print("Trained FrameHopper on", example_video)

    # 3) Prepare snippet dataset + dataloader for MIL
    # For quick demo use at most 8 videos
    small_list = files[:8]
    dataset = SnippetDataset(small_list, snippet_frames=16, resize=(320,240), max_frames=512)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    # 4) Train MIL model (very small epochs for demo)
    model = MIL_FCNet(in_dim=2048)
    train_mil(model, dataloader, epochs=2, device='cpu')
    torch.save(model.state_dict(), "mil_model.pth")
    print("Saved MIL model weights")

def demo_infer_single(video_path):
    # load framehopper
    from part4_framehopper import FrameHopper
    fh = FrameHopper()
    try:
        fh.load("framehopper_q.pkl")
    except:
        print("No saved FrameHopper found (framehopper_q.pkl). Run demo_train_small first.")
        return

    # read frames
    frames = read_video_frames(video_path, resize=(320,240), max_frames=600)
    sel_idxs = fh.inference_select_frames(frames)
    print(f"Selected {len(sel_idxs)} frames out of {len(frames)}")

    # group selected into snippets of 16 indices (drop remainder)
    snippets = []
    for i in range(0, len(sel_idxs), 16):
        idxs = sel_idxs[i:i+16]
        if len(idxs) < 16:
            break
        snippet_frames = [frames[j] for j in idxs]
        # create feature the same way SnippetDataset did (use same placeholder)
        import numpy as np, cv2
        avg = np.mean(snippet_frames, axis=0).astype('float32')/255.0
        tmp = avg.mean(axis=2)
        tmp = cv2.resize(tmp, (32,32)).astype('float32')
        vec = tmp.reshape(-1)
        feat = np.zeros(2048, dtype='float32')
        feat[:vec.shape[0]] = vec
        snippets.append(feat)
    if len(snippets) == 0:
        print("No full snippets formed from selected frames.")
        return
    feats = torch.from_numpy(np.stack(snippets, axis=0)).float()  # (S,2048)

    # load MIL model
    model = MIL_FCNet(in_dim=2048)
    try:
        model.load_state_dict(torch.load("mil_model.pth", map_location='cpu'))
    except:
        print("No saved mil_model.pth found. Run demo_train_small first.")
        return

    # score
    scores = score_video_model(model, feats, device='cpu')
    print("Snippet scores:", scores)
    anomalies = [i for i,s in enumerate(scores) if s>0.5]
    print("Detected anomaly snippet indices (threshold 0.5):", anomalies)

if __name__ == "__main__":
    # Example run
    videos_root = "videos"   # expect videos/normal and videos/anomaly
    demo_train_small(videos_root)
    # inference on a sample video placed at videos/test/example.mp4
    demo_infer_single(os.path.join("videos","test","example.mp4"))
