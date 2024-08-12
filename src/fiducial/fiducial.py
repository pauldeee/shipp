import os.path

import cv2


class Fiducial:
    def __init__(self, position: str, image):
        self.position = position
        self.image = image


class FiducialTemplate:

    def __init__(self, template_directory):
        self.template_directory = template_directory
        self.name = os.path.basename(template_directory)

        self.fiducials = self._load_fiducial_markers()

    def _load_fiducial_markers(self):
        markers = os.listdir(self.template_directory)
        fiducials = []

        for marker in markers:
            name = os.path.splitext(marker)[0]
            fiducials.append(
                Fiducial(name, cv2.imread(os.path.join(self.template_directory, marker), cv2.IMREAD_GRAYSCALE)))

        return fiducials


if __name__ == '__main__':
    ftpath = "../../templates/1958"

    ft = FiducialTemplate(ftpath)

    print(ft.name)

    print(ft.fiducials[0].position)
    print(ft.fiducials[1].position)
    print(ft.fiducials[2].position)
    print(ft.fiducials[3].position)
