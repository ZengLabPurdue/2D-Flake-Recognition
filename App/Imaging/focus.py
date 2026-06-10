import time
import cv2

class FocusController:
    def __init__(
        self,
        stage,
        frame_processor,
        disable_buttons=None,
        enable_buttons=None,
        sharpness_callback=None,
    ):
        self.stage = stage
        self.frame_processor = frame_processor

        self.disable_buttons = disable_buttons or (lambda: None)
        self.enable_buttons = enable_buttons or (lambda: None)
        self.sharpness_callback = sharpness_callback or (lambda sharpness: None)

    def find_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        self.sharpness_callback(sharpness)

        return sharpness

    def get_raw_sharpness(self, num_images=2):
        sharpness = 0

        for _ in range(num_images):
            image = self.frame_processor.capture_frame()
            sharpness += self.find_sharpness(image)

        return sharpness / num_images

    def discard_initial_frame(self, position):
        self.stage.move_to_z(position)
        self.stage.wait_until_not_busy()

        self.get_raw_sharpness(num_images=3)

    def find_best_focus(self, z_start, z_end, steps, tolerance=0.2):
        best_sharpness = -1
        best_z = z_start
        drops = 0

        z_positions = [
            z_start + i * (z_end - z_start) / steps
            for i in range(steps + 1)
        ]

        print(
            f"Speed: {self.stage.get_z_velocity()}, "
            f"Step: {int((z_end - z_start) / steps)}"
        )

        curr_z = self.stage.get_z_position()

        # Start from the side closest to current Z
        if abs(curr_z - z_positions[0]) < abs(curr_z - z_positions[-1]):
            z_positions.reverse()

        self.discard_initial_frame(z_positions[0])

        for z in z_positions:
            self.stage.move_to_z(z)
            self.stage.wait_until_not_busy()

            sharpness = self.get_raw_sharpness(num_images=3)

            print(
                f"Z: {z:>12.1f} | "
                f"Sharpness: {sharpness:>8.3f} | "
                f"Best Sharpness: {best_sharpness:>8.3f} | "
                f"Best Z: {best_z:>12.1f}"
            )

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_z = z
                drops = 0
            else:
                if sharpness < best_sharpness - tolerance:
                    drops += 1

            if drops >= 2:
                print("Focus peak passed")
                break

        self.stage.move_to_z(best_z)
        self.stage.wait_until_not_busy()

        return best_z

    def auto_focus(self, start_range=3000, accuracy=50, steps=20):
        start_time = time.time()

        self.disable_buttons()

        try:
            search_range = start_range
            best_z = self.stage.get_z_position()

            while search_range >= accuracy:
                best_z = self.find_best_focus(
                    best_z - search_range,
                    best_z + search_range,
                    steps
                )

                self.stage.move_to_z(best_z)
                self.stage.wait_until_not_busy()

                self.discard_initial_frame(best_z)

                sharpness = self.get_raw_sharpness(num_images=3)

                print(
                    f"Best Z: {best_z:>12.1f} | "
                    f"Sharpness: {sharpness:>8.3f} | "
                    f"Range: {search_range}"
                )
                print("-----------------------------------")

                search_range = int(search_range / (steps / 2))

            print(f"Time taken: {time.time() - start_time:.2f}s")

        finally:
            self.enable_buttons()