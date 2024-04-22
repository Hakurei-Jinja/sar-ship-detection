from io import TextIOWrapper
import xml.etree.ElementTree as ET


class XMLParser:
    def get_xml_root(self, file: TextIOWrapper) -> ET.Element:
        return ET.parse(file).getroot()

    def get_xml_element(self, element: ET.Element, tag: str) -> ET.Element:
        if element.find(tag) is None:
            raise ValueError(f"Tag {tag} not found")
        return element.find(tag)  # type: ignore

    def get_xml_text(self, element: ET.Element, tag: str) -> str:
        if element.find(tag) is None:
            raise ValueError(f"Tag {tag} not found")
        if element.find(tag).text is None:  # type: ignore
            raise ValueError(f"Tag {tag} text not found")
        return element.find(tag).text  # type: ignore

    def get_text(self, element: ET.Element) -> str:
        if element.text is None:
            raise ValueError("Text not found")
        return element.text

    def get_xml_iter(self, element: ET.Element, tag: str) -> ET.Element:
        return element.iter(tag)  # type: ignore
