# import unittest
import pytest
import Upload_Picture
import numpy as np
from PIL import Image
import PIL as PIL
from Predict import Predict


class TestPredict:
    image_name = 'from_camera'

    @pytest.fixture
    def image_path(self):
        return r'..\test_image.jpg'

    @pytest.fixture
    def image_taken(self):
        return Upload_Picture.take_picture(self.image_name)

    @pytest.fixture
    def PIL_from_path(self, image_path):
        return Image.open(image_path)

    @pytest.fixture
    def numpy_from_path(self, PIL_from_path):
        return np.array(PIL_from_path)

    def test_image_taken(self, image_taken):
        """
        Asserts the return value of  Upload_Picture.take_picture function
        :param image_taken: return value of Upload_Picture.take_picture function
        """
        assert  type( Image.open(image_taken[0])).__name__ == 'PngImageFile'
        assert type(image_taken[1]).__name__ == 'tuple'

    def test_get_mask_from_local_image(self, image_path):
        """
        Checks the return value of  Predict().get_mask_from_local_image function.
        :param image_path: path to the test_image.jpg
        """
        a, b = Predict().get_mask_from_local_image(image_path)
        assert isinstance( a, (PIL.Image.Image) )
        assert isinstance(b, (PIL.JpegImagePlugin.JpegImageFile ) )

    def test_get_mask_from_array(self, numpy_from_path):
        """
        Checks the return value of  Predict().get_mask_from_array function.
        :param numpy_from_path: numpy array representation of test_image.jpg
        """
        a = Predict().get_mask_from_array(numpy_from_path)
        assert isinstance( a, PIL.Image.Image)




