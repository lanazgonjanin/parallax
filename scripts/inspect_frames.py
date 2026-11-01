import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    image_dir = Path("data/sequences/00/image_0")

    frame_indices = [525, 535, 540, 543, 546, 550, 555, 560, 565, 575]  # frames surrounding the pose recovery failures
    #frame_indices = [450, 460, 470, 480, 490, 500, 510, 520, 530, 540]  # frames just before the failures

    fig, axes = plt.subplots(5, 2, figsize=(16, 12))

    for ax, frame_idx in zip(axes.flat, frame_indices):
        image_path = image_dir / f"{frame_idx:06d}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not load {image_path}")

        ax.imshow(image, cmap="gray")
        ax.set_title(f"Frame {frame_idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/frame_inspection.png", dpi=150)


if __name__ == "__main__":
    main()
