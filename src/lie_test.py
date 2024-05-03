import torch
from absl.testing import absltest
from lie import SO3, SE3


class TestSO3(absltest.TestCase):

    def test_so3_valid_input(self):
        # Valid SO3 matrix
        valid_so3 = torch.eye(3)
        so3_instance = SO3(valid_so3)
        self.assertTrue(
            torch.allclose(so3_instance.value.squeeze(), valid_so3),
            "SO3 instance not correctly formed with valid input")

    def test_so3_invalid_shape(self):
        # Invalid shape
        invalid_shape_so3 = torch.randn(3, 2)
        with self.assertRaises(ValueError):
            SO3(invalid_shape_so3)

    def test_so3_non_orthonormal(self):
        # Non-orthonormal matrix
        non_orthonormal = torch.tensor([[2., 0., 0.],
                                        [0., 1., 1.],
                                        [0., 1., 1.]])
        with self.assertRaises(ValueError):
            SO3(non_orthonormal)


class TestSE3(absltest.TestCase):

    def test_se3_valid_input(self):
        # Valid SE3 matrix
        valid_se3 = torch.eye(4)
        se3_instance = SE3(valid_se3)
        self.assertTrue(
            torch.allclose(se3_instance.value.squeeze(), valid_se3),
            "SE3 instance not correctly formed with valid input")

    def test_se3_invalid_shape(self):
        # Invalid shape
        invalid_shape_se3 = torch.randn(4, 3)
        with self.assertRaises(ValueError):
            SE3(invalid_shape_se3)

    def test_se3_non_orthonormal(self):
        # Non-orthonormal rotation part
        non_orthonormal = torch.tensor([[2., 0., 0., 0.],
                                        [0., 1., 1., 0.],
                                        [0., 1., 1., 0.],
                                        [0., 0., 0., 1.]])
        with self.assertRaises(ValueError):
            SE3(non_orthonormal)

    def test_se3_return_rt(self):
        # Check return_RT function
        valid_se3 = torch.eye(4)
        se3_instance = SE3(valid_se3)
        R, T = se3_instance.return_RT()
        self.assertTrue(torch.allclose(R, torch.eye(3)),
                        "Rotation part incorrect")
        self.assertTrue(torch.allclose(T, torch.zeros(3)),
                        "Translation part incorrect")


if __name__ == '__main__':
    absltest.main()
