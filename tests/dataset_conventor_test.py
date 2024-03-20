from io import TextIOWrapper
import os
import shutil
import sys
import unittest

sys.path.append(".")
from src.dataset.dataset_convertor import DatasetConvertor

correct_labels = {
    "000001.txt": [[0.0, 0.57933, 0.29721, 0.11538, 0.30341]],
    "000006.txt": [
        [0.0, 0.28144, 0.43978, 0.16367, 0.12325],
        [0.0, 0.73353, 0.46919, 0.10978, 0.05882],
        [0.0, 0.53593, 0.76611, 0.02595, 0.14286],
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


class TestDatasetConvertor(unittest.TestCase):
    def test_create_empty(self):
        dataset_convertor = DatasetConvertor()
        self.assertIsInstance(dataset_convertor, DatasetConvertor)
        self.assertEqual(dataset_convertor._config, [])

    def test_convert_config_empty_raise_exception(self):
        dataset_convertor = DatasetConvertor()
        with self.assertRaises(NameError):
            dataset_convertor.convert()

    def test_create_from_yaml(self):
        dataset_convertor = DatasetConvertor(
            "./tests/test_dataset/cfg/test_convertor_list.yaml"
        )
        self.assertIsInstance(dataset_convertor._config, list)
        self.assertEqual(len(dataset_convertor._config), 2)

    def test_load_config_from_yaml_list(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/test_convertor_list.yaml"
        )
        self.assertIsInstance(dataset_convertor._config, list)
        self.assertEqual(len(dataset_convertor._config), 2)

    def test_load_config_from_yaml_dict(self):
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/test_convertor_dict.yaml"
        )
        self.assertIsInstance(dataset_convertor._config, list)
        self.assertEqual(len(dataset_convertor._config), 1)

    def test_convert_images(self):
        rm_dir("./tests/test_dataset/converted")

        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/test_convertor_list.yaml"
        )
        dataset_convertor.convert()
        images01 = os.listdir("./tests/test_dataset/converted/01/images/train")
        images02 = os.listdir("./tests/test_dataset/converted/01/images/test")
        self.assertTrue("000001.jpg" in images01 and "000006.jpg" in images01)
        self.assertTrue("000002.jpg" in images02 and "000015.jpg" in images02)

    def test_convert_labels_exist(self):
        rm_dir("./tests/test_dataset/converted")
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/test_convertor_list.yaml"
        )
        dataset_convertor.convert()
        labels01 = os.listdir("./tests/test_dataset/converted/01/labels/train")
        labels02 = os.listdir("./tests/test_dataset/converted/01/labels/test")
        self.assertTrue("000001.txt" in labels01 and "000006.txt" in labels01)
        self.assertTrue("000002.txt" in labels02 and "000015.txt" in labels02)

    def test_convert_labels_correct(self):
        rm_dir("./tests/test_dataset/converted")
        dataset_convertor = DatasetConvertor()
        dataset_convertor.load_config(
            "./tests/test_dataset/cfg/test_convertor_list.yaml"
        )
        dataset_convertor.convert()
        with open(
            "./tests/test_dataset/converted/01/labels/train/000001.txt", "r"
        ) as f:
            converted = convert_label_txt(f)
            self.assertEqual(converted, correct_labels["000001.txt"])
        with open(
            "./tests/test_dataset/converted/01/labels/train/000006.txt", "r"
        ) as file:
            converted = convert_label_txt(file)
            self.assertEqual(converted, correct_labels["000006.txt"])


if __name__ == "__main__":
    unittest.main()
