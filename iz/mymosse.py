import cv2
import numpy as np


class MOSSETrackerFromOzon:
    def __init__(self, learning_rate=0.2, psr_threshold=12.0):
        self.lr = learning_rate
        self.psr_threshold = psr_threshold

        self.H = None
        self.G = None
        self.win = None
        self.pos = None
        self.size = None

        self.smooth_factor = 0.6

    def preprocess(self, img):
        img = np.log(img + 1)
        img = (img - img.mean()) / (img.std() + 1e-5)
        return img * self.win

    def gaus2D(self, size):
        w, h = size
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        cx, cy = w // 2, h // 2
        sigma = 0.125 * max(w, h)
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        return g

    def get_samples(self, img, count=2):
        samples = []
        for _ in range(count):
            noise = (np.random.randn(*img.shape) * 5).astype(np.float32)
            bright = img + np.random.uniform(-5, 5)
            aug = np.clip(bright + noise, 0, 255)
            samples.append(aug.astype(np.float32))
        return samples

    def init(self, bbox, frame):
        x, y, w, h = bbox
        self.size = (w, h)
        self.pos = (x + w / 2, y + h / 2)

        patch = cv2.getRectSubPix(frame, (w, h), self.pos)
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)

        self.win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

        self.G = self.gaus2D((w, h))
        G_fft = np.fft.fft2(self.G)

        imgs = [patch_gray] + self.get_samples(patch_gray)

        A = 0
        B = 0

        for img in imgs:
            F = np.fft.fft2(self.preprocess(img))
            A += G_fft * np.conj(F)
            B += F * np.conj(F)

        self.H = A / (B + 1e-5)

    def update(self, frame):
        w, h = self.size
        cx, cy = self.pos

        patch = cv2.getRectSubPix(frame, (w, h), (cx, cy))
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)

        F = np.fft.fft2(self.preprocess(patch_gray))
        R = np.fft.ifft2(F * self.H)
        R = np.real(R)

        max_loc = np.unravel_index(np.argmax(R), R.shape)
        dy = max_loc[0] - h // 2
        dx = max_loc[1] - w // 2

        peak = R[max_loc]
        side = R.copy()
        side[max_loc[0] - 3 : max_loc[0] + 3, max_loc[1] - 3 : max_loc[1] + 3] = 0

        psr = (peak - side.mean()) / (side.std() + 1e-5)

        new_cx = cx + dx
        new_cy = cy + dy

        new_cx = (1 - self.smooth_factor) * cx + self.smooth_factor * new_cx
        new_cy = (1 - self.smooth_factor) * cy + self.smooth_factor * new_cy

        H_img, W_img = frame.shape[:2]
        new_cx = np.clip(new_cx, w / 2, W_img - w / 2)
        new_cy = np.clip(new_cy, h / 2, H_img - h / 2)

        self.pos = (float(new_cx), float(new_cy))

        if psr > self.psr_threshold:
            G_fft = np.fft.fft2(self.G)
            F_new = np.fft.fft2(self.preprocess(patch_gray))
            H_new = (G_fft * np.conj(F_new)) / (F_new * np.conj(F_new) + 1e-5)
            self.H = (1 - self.lr) * self.H + self.lr * H_new

        x = int(self.pos[0] - w / 2)
        y = int(self.pos[1] - h / 2)
        return x, y, w, h
