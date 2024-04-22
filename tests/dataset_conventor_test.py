from io import TextIOWrapper
import os
import shutil
import sys
import unittest

sys.path.append(".")
from src.dataset.dataset_convertor import DatasetConvertor

correct_train_imgs = ["000001.jpg", "000006.jpg"]
correct_test_imgs = ["000002.jpg", "000015.jpg"]
correct_train_labels = ["000001.txt", "000006.txt"]
correct_test_labels = ["000002.txt", "000015.txt"]

voc_correct_labels_value = {
    "000001.txt": [[0.0, 0.57933, 0.29721, 0.11538, 0.30341]],
    "000006.txt": [
        [0.0, 0.28144, 0.43978, 0.16367, 0.12325],
        [0.0, 0.73353, 0.46919, 0.10978, 0.05882],
        [0.0, 0.53593, 0.76611, 0.02595, 0.14286],
    ],
}

voc_obb_correct_labels_value = {
    "000001.txt": [
        [0.0, 0.51683, 0.14861, 0.6274, 0.13932, 0.64423, 0.44272, 0.53606, 0.45511]
    ],
    "000006.txt": [
        [0.0, 0.19361, 0.46499, 0.35529, 0.36695, 0.37126, 0.41737, 0.20758, 0.51541],
        [0.0, 0.67665, 0.47619, 0.68263, 0.43697, 0.79042, 0.46499, 0.78643, 0.5042],
        [0.0, 0.52295, 0.83754, 0.52695, 0.69468, 0.5509, 0.69748, 0.54691, 0.84034],
    ],
}

voc_seg_correct_labels_value = {
    # fmt: off
    "000001.txt": [
        [ 0.0, 0.54327, 0.22291, 0.53846, 0.17647, 0.54567, 0.14861,
          0.5649, 0.1517, 0.57212, 0.17647, 0.58654, 0.2291, 0.61298,
          0.2322, 0.62019, 0.28173, 0.63942, 0.32817, 0.61779, 0.40867,
          0.60577, 0.44892, 0.57452, 0.45201, 0.5625, 0.42415, 0.54567,
          0.39319, 0.53125, 0.36533, 0.53125, 0.31269, 0.52404, 0.23839]
    ],
    "000006.txt": [
        [0.0, 0.2016, 0.47059, 0.33533, 0.38095, 0.36527, 0.39496,
         0.35529, 0.42857, 0.22954, 0.5042, 0.20359, 0.4986],
        [0.0, 0.68064, 0.45938, 0.69661, 0.44258, 0.78643, 0.46499,
         0.79042, 0.47899, 0.77645, 0.5014, 0.70858, 0.48459],
        [0.0, 0.53693, 0.69748, 0.52495, 0.82073, 0.5489, 0.84034, 0.5509, 0.71989],
    ],
    # fmt: on
}

coco_correct_labels_value = {
    "000001.txt": [[0.0, 0.58333, 0.30134, 0.11538, 0.30547]],
    "000006.txt": [
        [0.0, 0.28452, 0.44519, 0.16229, 0.12304],
        [0, 0.73595, 0.47334, 0.11006, 0.05628],
        [0, 0.53915, 0.76916, 0.02612, 0.14267],
    ],
}
coco_remap_labels_value = {
    "000001.txt": [[1.0, 0.58333, 0.30134, 0.11538, 0.30547]],
    "000006.txt": [
        [1.0, 0.28452, 0.44519, 0.16229, 0.12304],
        [1.0, 0.73595, 0.47334, 0.11006, 0.05628],
        [1.0, 0.53915, 0.76916, 0.02612, 0.14267],
    ],
}


def rm_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def convert_label_txt(file: TextIOWrapper):
    converted = []
    contents = file.read().strip().split("\n")
    for content in contents:
        content = content.split(" ")
        converted.append([round(float(x), 5) for x in content])
    return converted


def file_list_equal(file_list1, file_list2):
    file_list1.sort()
    file_list2.sort()
    return file_list1 == file_list2


class TestDatasetConvertor(unittest.TestCase):
    def setUp(self):
        rm_dir("./tests/test_dataset/converted")

    def tearDown(self):
        rm_dir("./tests/test_dataset/converted")

    # general test
    def test_create_empty(self):
        dataset_convertor = DatasetConvertor()
        self.assertIsInstance(dataset_convertor, DatasetConvertor)
        self.assertEqual(dataset_convertor._config, [])

    def test_create_from_yaml(self):
        dataset_convertor = DatasetConvertor(
            "./tests/test_dataset/cfg/VOC/test_convertor_list.yaml"
        )
        self.assertIsInstance(dataset_convertor._config, list)
        self.assertEqual(len(dataset_convertor._config), 2)

    def test_load_config_from_yaml_list(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOC/test_convertor_list.yaml"
        )
        self.assertIsInstance(dataset_convertor._config, list)
        self.assertEqual(len(dataset_convertor._config), 2)

    def test_load_config_from_yaml_dict(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOC/test_convertor_dict.yaml"
        )
        self.assertIsInstance(dataset_convertor._config, list)
        self.assertEqual(len(dataset_convertor._config), 1)

    def test_convert_config_empty_raise_exception(self):
        dataset_convertor = DatasetConvertor()
        with self.assertRaises(NameError):
            dataset_convertor.convert()

    # VOC test
    def test_convert_voc_without_class_file_raise_exception(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOC/test_convertor_list.yaml"
        )
        with self.assertRaises(NameError):
            dataset_convertor.convert()

    def test_convert_voc_images(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOC/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        train_imgs = os.listdir("./tests/test_dataset/converted/VOC/01/images/train")
        test_imgs = os.listdir("./tests/test_dataset/converted/VOC/01/images/test")
        self.assertTrue(file_list_equal(train_imgs, correct_train_imgs))
        self.assertTrue(file_list_equal(test_imgs, correct_test_imgs))

    def test_convert_voc_labels_exist(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOC/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        train_labels = os.listdir("./tests/test_dataset/converted/VOC/01/labels/train")
        test_labels = os.listdir("./tests/test_dataset/converted/VOC/01/labels/test")
        self.assertTrue(file_list_equal(train_labels, correct_train_labels))
        self.assertTrue(file_list_equal(test_labels, correct_test_labels))

    def test_convert_voc_labels_correct(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOC/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        with open(
            "./tests/test_dataset/converted/VOC/01/labels/train/000001.txt", "r"
        ) as f:
            converted = convert_label_txt(f)
            self.assertEqual(converted, voc_correct_labels_value["000001.txt"])
        with open(
            "./tests/test_dataset/converted/VOC/01/labels/train/000006.txt", "r"
        ) as file:
            converted = convert_label_txt(file)
            self.assertEqual(converted, voc_correct_labels_value["000006.txt"])

    # VOCOBB test
    def test_convert_voc_obb_without_class_file_raise_exception(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCOBB/test_convertor_list.yaml"
        )
        with self.assertRaises(NameError):
            dataset_convertor.convert()

    def test_convert_voc_obb_images(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCOBB/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        train_imgs = os.listdir("./tests/test_dataset/converted/VOCOBB/01/images/train")
        test_imgs = os.listdir("./tests/test_dataset/converted/VOCOBB/01/images/test")
        self.assertTrue(file_list_equal(train_imgs, correct_train_imgs))
        self.assertTrue(file_list_equal(test_imgs, correct_test_imgs))

    def test_convert_voc_obb_labels_exist(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCOBB/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        train_labels = os.listdir(
            "./tests/test_dataset/converted/VOCOBB/01/labels/train"
        )
        test_labels = os.listdir("./tests/test_dataset/converted/VOCOBB/01/labels/test")
        self.assertTrue(file_list_equal(train_labels, correct_train_labels))
        self.assertTrue(file_list_equal(test_labels, correct_test_labels))

    def test_convert_voc_obb_labels_correct(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCOBB/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        with open(
            "./tests/test_dataset/converted/VOCOBB/01/labels/train/000001.txt", "r"
        ) as f:
            converted = convert_label_txt(f)
            self.assertEqual(converted, voc_obb_correct_labels_value["000001.txt"])
        with open(
            "./tests/test_dataset/converted/VOCOBB/01/labels/train/000006.txt", "r"
        ) as file:
            converted = convert_label_txt(file)
            self.assertEqual(converted, voc_obb_correct_labels_value["000006.txt"])

    # VOCSEG test
    def test_convert_voc_seg_without_class_file_raise_exception(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCSEG/test_convertor_list.yaml"
        )
        with self.assertRaises(NameError):
            dataset_convertor.convert()

    def test_convert_voc_seg_images(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCSEG/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        train_imgs = os.listdir("./tests/test_dataset/converted/VOCSEG/01/images/train")
        test_imgs = os.listdir("./tests/test_dataset/converted/VOCSEG/01/images/test")
        self.assertTrue(file_list_equal(train_imgs, correct_train_imgs))
        self.assertTrue(file_list_equal(test_imgs, correct_test_imgs))

    def test_convert_voc_seg_labels_exist(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCSEG/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        train_labels = os.listdir(
            "./tests/test_dataset/converted/VOCSEG/01/labels/train"
        )
        test_labels = os.listdir("./tests/test_dataset/converted/VOCSEG/01/labels/test")
        self.assertTrue(file_list_equal(train_labels, correct_train_labels))
        self.assertTrue(file_list_equal(test_labels, correct_test_labels))

    def test_convert_voc_seg_labels_correct(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/VOCSEG/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/class.yaml",
        )
        dataset_convertor.convert()
        with open(
            "./tests/test_dataset/converted/VOCSEG/01/labels/train/000001.txt", "r"
        ) as f:
            converted = convert_label_txt(f)
            self.assertEqual(converted, voc_seg_correct_labels_value["000001.txt"])
        with open(
            "./tests/test_dataset/converted/VOCSEG/01/labels/train/000006.txt", "r"
        ) as file:
            converted = convert_label_txt(file)
            self.assertEqual(converted, voc_seg_correct_labels_value["000006.txt"])

    # COCO test
    def test_convert_coco_images(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/coco/test_convertor_list.yaml"
        )
        dataset_convertor.convert()
        train_imgs = os.listdir("./tests/test_dataset/converted/coco/01/images/train")
        test_imgs = os.listdir("./tests/test_dataset/converted/coco/01/images/test")
        self.assertTrue(file_list_equal(train_imgs, correct_train_imgs))
        self.assertTrue(file_list_equal(test_imgs, correct_test_imgs))

    def test_convert_coco_labels_exist(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/coco/test_convertor_list.yaml"
        )
        dataset_convertor.convert()
        train_labels = os.listdir("./tests/test_dataset/converted/coco/01/labels/train")
        test_labels = os.listdir("./tests/test_dataset/converted/coco/01/labels/test")
        self.assertTrue(file_list_equal(train_labels, correct_train_labels))
        self.assertTrue(file_list_equal(test_labels, correct_test_labels))

    def test_convert_coco_labels_correct(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/coco/test_convertor_list.yaml",
        )
        dataset_convertor.convert()
        with open(
            "./tests/test_dataset/converted/coco/01/labels/train/000001.txt", "r"
        ) as f:
            converted = convert_label_txt(f)
            self.assertEqual(converted, coco_correct_labels_value["000001.txt"])
        with open(
            "./tests/test_dataset/converted/coco/01/labels/train/000006.txt", "r"
        ) as file:
            converted = convert_label_txt(file)
            self.assertEqual(converted, coco_correct_labels_value["000006.txt"])

    def test_convert_coco_remap_labels_correct(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/coco/test_convertor_list.yaml",
            "./tests/test_dataset/cfg/remap_class.yaml",
        )
        dataset_convertor.convert()
        with open(
            "./tests/test_dataset/converted/coco/01/labels/train/000001.txt", "r"
        ) as f:
            converted = convert_label_txt(f)
            self.assertEqual(converted, coco_remap_labels_value["000001.txt"])
        with open(
            "./tests/test_dataset/converted/coco/01/labels/train/000006.txt", "r"
        ) as file:
            converted = convert_label_txt(file)
            self.assertEqual(converted, coco_remap_labels_value["000006.txt"])


if __name__ == "__main__":
    unittest.main()
