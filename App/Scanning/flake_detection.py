import cv2
from Scanning.flake_identifier import Flake_Identifier

class Flake_Detector:

    def __init__(self):
        
        self.flake_identifier = Flake_Identifier()

    def flake_detection(self, image_queue, frame_processor):
        while True:
            img_path = image_queue.get()

            if img_path is None:
                break

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is None:
                continue

            scanned_img, _, save = self.flake_identifier.identify_flakes_flake_model(img)

            out_path = img_path.parent.parent / "Processed" / img_path.name

            frame_processor.save_image(
                cv2.cvtColor(scanned_img, cv2.COLOR_RGB2BGR),
                save_dir=out_path.parent,
                filename=out_path.name
            )

            if save:
                chip_folder = img_path.parent.parent
                scan_root = chip_folder.parent.parent.parent

                flakes_dir = scan_root / "Flakes Found" / chip_folder.name
                flakes_dir.mkdir(parents=True, exist_ok=True)

                frame_processor.save_image(
                    cv2.cvtColor(scanned_img, cv2.COLOR_RGB2BGR),
                    save_dir=flakes_dir,
                    filename=img_path.name
                )

            image_queue.task_done()