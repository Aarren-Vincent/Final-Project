# part4_framehopper.py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import random
import pickle

class FrameHopper:
    def __init__(self, n_states=10, max_skip=6, theta=0.15, alpha=0.1, gamma=0.9, psi1=1.0, psi2=1.0):
        self.n_states = n_states
        self.max_skip = max_skip
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma
        self.psi1 = psi1
        self.psi2 = psi2
        self.qtable = np.zeros((n_states, max_skip+1), dtype=np.float32)
        self.kmeans = MiniBatchKMeans(n_clusters=n_states, batch_size=64)
        self.fitted_kmeans = False

    def frame_change_vector(self, f_prev, f_cur, thresh=20):
        H, W, _ = f_prev.shape
        h_steps = np.linspace(0, H, 4, dtype=int)
        w_steps = np.linspace(0, W, 4, dtype=int)
        vec = []
        for i in range(3):
            for j in range(3):
                r0, r1 = h_steps[i], h_steps[i+1]
                c0, c1 = w_steps[j], w_steps[j+1]
                prev_chunk = f_prev[r0:r1, c0:c1].astype(int)
                cur_chunk = f_cur[r0:r1, c0:c1].astype(int)
                diff = np.abs(prev_chunk - cur_chunk)
                frac = (np.sum(np.any(diff>thresh, axis=2)) / (diff.shape[0]*diff.shape[1] + 1e-9))
                vec.append(frac)
        return np.array(vec)

    def fit_state_clusters(self, list_of_frame_lists, n_iter=10):
        # list_of_frame_lists is [frames_video1, frames_video2, ...]
        X = []
        for frames in list_of_frame_lists:
            for i in range(1, len(frames)):
                X.append(self.frame_change_vector(frames[i-1], frames[i]))
        X = np.stack(X, axis=0)
        # partial_fit in chunks
        self.kmeans.partial_fit(X)
        self.fitted_kmeans = True

    def get_state(self, f_prev, f_cur):
        v = self.frame_change_vector(f_prev, f_cur).reshape(1, -1)
        if not self.fitted_kmeans:
            return int(min(self.n_states-1, np.sum(v > 0.05)))
        return int(self.kmeans.predict(v)[0])

    def approx_distance(self, f_ref, f_target):
        # quick proxy: mean absolute pixel diff normalized 0..1
        return np.mean(np.abs(f_ref.astype('float32') - f_target.astype('float32'))) / 255.0

    def compute_reward(self, d, k):
        if d <= self.theta:
            return self.psi1 * (k+1)
        else:
            return -self.psi2 * k

    def train(self, frames, epochs=1, epsilon=0.2):
        if not self.fitted_kmeans:
            self.fit_state_clusters([frames])
        for _ in range(epochs):
            idx = 0
            f_prev = frames[idx]
            while idx < len(frames)-1:
                f_cur = frames[min(idx+1, len(frames)-1)]
                s = self.get_state(f_prev, f_cur)
                if random.random() < epsilon:
                    a = random.randint(0, self.max_skip)
                else:
                    a = int(np.argmax(self.qtable[s]))
                next_idx = min(idx + 1 + a, len(frames)-1)
                d = self.approx_distance(f_prev, frames[next_idx])
                r = self.compute_reward(d, a)
                if next_idx < len(frames)-1:
                    s_next = self.get_state(frames[next_idx], frames[min(next_idx+1, len(frames)-1)])
                else:
                    s_next = s
                a_next = int(np.argmax(self.qtable[s_next]))
                old = self.qtable[s, a]
                self.qtable[s, a] = old + self.alpha * (r + self.gamma * self.qtable[s_next, a_next] - old)
                idx = next_idx
                f_prev = frames[idx]

    def inference_select_frames(self, frames):
        selected = [0]
        idx = 0
        f_prev = frames[0]
        while idx < len(frames)-1:
            s = self.get_state(f_prev, frames[min(idx+1, len(frames)-1)])
            a = int(np.argmax(self.qtable[s]))
            next_idx = min(idx + 1 + a, len(frames)-1)
            selected.append(next_idx)
            idx = next_idx
            f_prev = frames[idx]
        return selected

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'qtable': self.qtable, 'kmeans': self.kmeans}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.qtable = d['qtable']
        self.kmeans = d['kmeans']
        self.fitted_kmeans = True
