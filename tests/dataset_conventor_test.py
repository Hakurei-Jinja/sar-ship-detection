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
