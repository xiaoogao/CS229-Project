import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class ManualFaceMasker:
    def __init__(self, img_dir, output_dir):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.idx = 0
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.masked_img = None  # For storing the currently masked version

    def run(self):
        while 0 <= self.idx < len(self.img_paths):
            img_path = self.img_paths[self.idx]
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            self.masked_img = None

            fig, ax = plt.subplots()
            ax.imshow(img_np)
            ax.set_title(f"{os.path.basename(img_path)} ({self.idx+1}/{len(self.img_paths)})\n"
                         "n=next, p=prev, m=mask, s=save, q=quit")

            def on_key(event):
                if event.key == 'q':
                    plt.close()
                    self.idx = len(self.img_paths)  # Break loop
                elif event.key == 'n':
                    plt.close()
                    self.idx += 1
                elif event.key == 'p':
                    plt.close()
                    self.idx = max(self.idx - 1, 0)
                elif event.key == 'm':
                    # Activate manual mask with two clicks
                    plt.close()
                    self.manual_mask(img_np, img_path)
                elif event.key == 's':
                    if self.masked_img is not None:
                        out_path = os.path.join(self.output_dir, os.path.basename(img_path))
                        Image.fromarray(self.masked_img).save(out_path)
                        print(f"[Saved] {out_path}")
                    else:
                        print("No mask to save. Press 'm' to mask first.")

            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()

    def manual_mask(self, img_np, img_path):
        # Let user click two points to draw rectangle mask
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        ax.set_title("Click left-top, then right-bottom. Close when done.")
        pts = plt.ginput(2, timeout=0)
        plt.close()
        if len(pts) != 2:
            print("Not enough points clicked, returning to viewer.")
            return
        (x1, y1), (x2, y2) = pts
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        mask_img = img_np.copy()
        mask_img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2), :] = 0

        # Show preview and set for save
        fig2, ax2 = plt.subplots()
        ax2.imshow(mask_img)
        ax2.set_title("Preview masked image. Close window to return. Press 's' in main viewer to save.")
        plt.show()
        self.masked_img = mask_img

if __name__ == "__main__":
    input_dir = "./val"               # Folder to review & mask
    output_dir = "./val_face_masked"  # Output folder for manually masked images

    masker = ManualFaceMasker(input_dir, output_dir)
    masker.run()

