"""
奖励函数：评估 SVG 徽标质量
维度：有效性、结构、坐标、颜色、关键词
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional

class SVGReward:    
    def __init__(self):
        self.weights = {
            "valid_xml": 0.25,
            "required_attrs": 0.20,
            "coordinate_bounds": 0.20,
            "color_diversity": 0.15,
            "keyword_match": 0.20,
        }
        self.viewbox_target = "0 0 256 256"
        # 常用颜色
        self.color_words = {
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "white", "black", "gray", "grey", "teal", "navy", "gold", "silver",
            "coral", "salmon", "crimson", "amber", "ivory", "cream"
        }
        # 形状关键词
        self.shape_map = {
            "circle": "circle", "round": "circle",
            "square": "rect", "rectangle": "rect",
            "triangle": "polygon", "star": "polygon",
            "line": "line", "path": "path"
        }
    
    def parse_svg(self, svg_text: str) -> Optional[ET.Element]:
        
        try:
            root = ET.fromstring(svg_text)
            return root
        except ET.ParseError:
            if "<svg" in svg_text and "</svg>" not in svg_text:
                svg_text += "</svg>"
            try:
                return ET.fromstring(svg_text)
            except:
                return None
    
    def check_valid_xml(self, root: Optional[ET.Element]) -> float:
        """检查 XML 是否有效"""
        return 1.0 if root is not None else 0.0
    
    def check_required_attrs(self, root: ET.Element, svg_text: str) -> float:
        """检查 xmlns 和 viewBox"""
        has_viewbox = "viewBox" in root.attrib
        has_xmlns = "xmlns" in svg_text or root.tag.startswith("{")
        if not has_viewbox or not has_xmlns:
            return 0.0
        # viewBox 值是否准确
        actual_viewbox = " ".join(root.attrib.get("viewBox", "").split())
        if actual_viewbox != self.viewbox_target:
            return 0.5  # 部分得分
        return 1.0
    
    def check_coordinate_bounds(self, root: ET.Element) -> float:
        """提取所有坐标，检查是否在合理范围（-50~350）"""
        coords = []
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            # 提取数值属性
            for attr in ["cx", "cy", "x", "y", "width", "height", "r", "rx", "ry"]:
                val = elem.attrib.get(attr)
                if val:
                    try:
                        coords.append(float(val))
                    except:
                        pass
            # path 的 d 属性
            if tag == "path":
                d = elem.attrib.get("d", "")
                nums = re.findall(r'[-+]?\d*\.?\d+', d)
                coords.extend([float(n) for n in nums])
            # polygon/polyline 的 points
            if tag in ["polygon", "polyline"]:
                pts = elem.attrib.get("points", "")
                for p in re.findall(r'[-+]?\d*\.?\d+', pts):
                    coords.append(float(p))
        
        if not coords:
            return 0.0
        min_c, max_c = min(coords), max(coords)
        # 容忍范围 -50 ~ 350
        if min_c >= -50 and max_c <= 350:
            return 1.0
        else:
            # 惩罚超出程度
            penalty = max(0, (abs(min_c) - 50) / 100) + max(0, (max_c - 350) / 100)
            return max(0.0, 1.0 - penalty)
    
    def check_color_diversity(self, root: ET.Element) -> float:
        """统计 fill/stroke 中的不同颜色数量，至少2种才给满分"""
        colors = set()
        for elem in root.iter():
            for attr in ["fill", "stroke"]:
                val = elem.attrib.get(attr)
                if val and val not in ["none", "transparent"]:
                    if val.startswith("url("):
                        continue
                    colors.add(val.lower())
        count = len(colors)
        if count >= 3:
            return 1.0
        elif count == 2:
            return 0.8
        elif count == 1:
            return 0.5
        else:
            return 0.0
    
    def check_keyword_match(self, root: ET.Element, prompt: str) -> float:
        """检查提示词中的颜色词和形状词是否出现在 SVG 中"""
        if not prompt:
            return 0.5
        prompt_lower = prompt.lower()
        svg_text = ET.tostring(root, encoding='unicode').lower()
        
        matches = 0
        total = 0
        
        # 颜色词
        for color in self.color_words:
            if color in prompt_lower:
                total += 1
                if color in svg_text:
                    matches += 1
        
        # 形状词
        for shape, tag in self.shape_map.items():
            if shape in prompt_lower:
                total += 1
                if f"<{tag}" in svg_text:
                    matches += 1
        
        if total == 0:
            return 0.5
        return matches / total
    
    def compute(self, svg_text: str, prompt: str = "") -> Dict:
        """返回总分数和各维度得分"""
        root = self.parse_svg(svg_text)
        
        scores = {}
        scores["valid_xml"] = self.check_valid_xml(root)
        
        if root is None:
            # 如果解析失败，其他维度为0
            for k in ["required_attrs", "coordinate_bounds", "color_diversity", "keyword_match"]:
                scores[k] = 0.0
        else:
            scores["required_attrs"] = self.check_required_attrs(root, svg_text)
            scores["coordinate_bounds"] = self.check_coordinate_bounds(root)
            scores["color_diversity"] = self.check_color_diversity(root)
            scores["keyword_match"] = self.check_keyword_match(root, prompt)
        
        # 加权总分
        total = sum(scores[k] * self.weights[k] for k in self.weights)
        
        return {
            "total": round(total, 4),
            "valid": root is not None,
            "breakdown": {k: round(v, 4) for k, v in scores.items()}
        }


def compute_reward(svg_text: str, prompt: str = "") -> Dict:
    """外部调用接口"""
    return SVGReward().compute(svg_text, prompt)