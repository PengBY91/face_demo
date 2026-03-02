"""
LLM 服务模块
LLM Service Module

提供自然语言查询解析功能，将用户输入转化为结构化查询条件
Parses natural language queries into structured query conditions
"""
import os
import re
import json
import requests
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class QueryCondition:
    """查询条件数据类"""
    person_names: List[str] = field(default_factory=list)  # 支持多个人员
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    # 支持多个时间段，每个时间段是 (start_time, end_time) 元组
    time_periods: List[tuple] = field(default_factory=list)
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    limit: int = 50
    offset: int = 0

    # 兼容旧接口
    @property
    def person_name(self) -> Optional[str]:
        return self.person_names[0] if self.person_names else None

    @property
    def start_time(self) -> Optional[str]:
        return self.time_periods[0][0] if self.time_periods else None

    @property
    def end_time(self) -> Optional[str]:
        return self.time_periods[0][1] if self.time_periods else None


class LLMService:
    """LLM 服务类，用于解析自然语言查询"""

    def __init__(self, config_path: str = "llm_config.ini"):
        self.config = self._load_config(config_path)
        self.base_url = self.config.get('base_url', '')
        self.api_key = self.config.get('api_key', '')
        self.model = self.config.get('model', 'deepseek-chat')
        self.enabled = bool(self.base_url and self.api_key)

    def _load_config(self, config_path: str) -> Dict:
        """加载 LLM 配置"""
        config = {}
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        line = line.strip()
                        if ':' in line and not line.startswith('#'):
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            config[key] = value
        except Exception as e:
            print(f"LLMService: 加载配置失败: {e}")
        return config

    def parse_query(self, query: str, cameras: List[Dict] = None) -> QueryCondition:
        """
        解析自然语言查询，返回结构化查询条件

        Args:
            query: 用户的自然语言查询
            cameras: 可用的摄像头列表 [{'id': 'xxx', 'name': 'xxx'}, ...]

        Returns:
            QueryCondition 对象

        Raises:
            Exception: 当 LLM 不可用时抛出异常
        """
        if not self.enabled:
            raise Exception("LLM 服务未配置，请在检查 llm_config.ini 中的 base_url 和 api_key 配置")

        result = self._llm_parse(query, cameras)
        return result

    def _rule_based_parse(self, query: str, cameras: List[Dict] = None) -> QueryCondition:
        """
        基于规则的自然语言解析（不依赖 LLM）
        """
        condition = QueryCondition()
        query_lower = query.lower()

        # 常见关键词列表（用于过滤）
        stop_words = ['昨天', '今天', '前天', '最近', '所有', '全部', '今天', '昨日',
                      '本周', '上周', '本月', '记录', '到访', '来访', '查询',
                      '搜索', '查找', '检索', '显示', '列出', '找出', '看看', '和', '与', '及']

        # 解析多个姓名
        # 模式: "搜索张三和李四的记录", "查询张三、李四、王五"
        names = []

        # 尝试匹配多个名字（用 和/与/及/、/， 分隔）
        multi_name_patterns = [
            r'(?:搜索|查询|查找|检索|找|看看|显示|列出)\s*([^\s的来去过全最本周和与及、，]{2,4}(?:\s*(?:和|与|及|、|，)\s*[^\s的来去过全最本周和与及、，]{2,4})+)',
        ]

        for pattern in multi_name_patterns:
            match = re.search(pattern, query)
            if match:
                name_str = match.group(1)
                # 分割名字
                potential_names = re.split(r'\s*(?:和|与|及|、|，)\s*', name_str)
                for n in potential_names:
                    if n and n not in stop_words and len(n) >= 2:
                        names.append(n)
                break

        # 如果没有找到多个名字，尝试单个名字
        if not names:
            single_name_patterns = [
                r'(?:搜索|查询|查找|检索|找|看看|显示|列出)\s*([^\s的来去过全最本周]{2,4})\s*(?:的|到访|来访|记录|的记录|的所有记录)',
                r'(?:搜索|查询|查找|检索|找|看看|显示|列出)\s*([^\s的来去过全最本周]{2,4})',
                r'([^\s的来去过全最本周]{2,4})\s*(?:的记录|到访|来访|出现过)',
            ]
            for pattern in single_name_patterns:
                match = re.search(pattern, query)
                if match:
                    potential_name = match.group(1)
                    if potential_name not in stop_words and len(potential_name) >= 2:
                        names.append(potential_name)
                        break

        condition.person_names = names

        # 解析多个时间段
        today = datetime.now()
        time_periods = []

        # 检查是否包含多个时间关键词
        time_keywords = {
            '今天': (0, 0),
            '今日': (0, 0),
            '昨天': (1, 1),
            '昨日': (1, 1),
            '前天': (2, 2),
        }

        found_time_keywords = []
        for keyword in time_keywords:
            if keyword in query:
                found_time_keywords.append(keyword)

        if len(found_time_keywords) > 1:
            # 多个时间段
            for keyword in found_time_keywords:
                days_ago_start, days_ago_end = time_keywords[keyword]
                start = today - timedelta(days=days_ago_start)
                end = today - timedelta(days=days_ago_end)
                time_periods.append((
                    start.strftime("%Y-%m-%d 00:00:00"),
                    end.strftime("%Y-%m-%d 23:59:59")
                ))
        elif '今天' in query or '今日' in query:
            time_periods.append((today.strftime("%Y-%m-%d 00:00:00"),
                                today.strftime("%Y-%m-%d 23:59:59")))
        elif '昨天' in query or '昨日' in query:
            yesterday = today - timedelta(days=1)
            time_periods.append((yesterday.strftime("%Y-%m-%d 00:00:00"),
                                yesterday.strftime("%Y-%m-%d 23:59:59")))
        elif '前天' in query:
            day_before = today - timedelta(days=2)
            time_periods.append((day_before.strftime("%Y-%m-%d 00:00:00"),
                                day_before.strftime("%Y-%m-%d 23:59:59")))
        elif '本周' in query:
            week_start = today - timedelta(days=today.weekday())
            time_periods.append((week_start.strftime("%Y-%m-%d 00:00:00"),
                                today.strftime("%Y-%m-%d 23:59:59")))
        elif '上周' in query:
            week_start = today - timedelta(days=today.weekday() + 7)
            week_end = week_start + timedelta(days=6)
            time_periods.append((week_start.strftime("%Y-%m-%d 00:00:00"),
                                week_end.strftime("%Y-%m-%d 23:59:59")))
        elif '本月' in query:
            time_periods.append((today.strftime("%Y-%m-01 00:00:00"),
                                today.strftime("%Y-%m-%d 23:59:59")))
        elif '最近' in query:
            match = re.search(r'最近\s*(\d+)\s*天', query)
            days = int(match.group(1)) if match else 7
            start = today - timedelta(days=days)
            time_periods.append((start.strftime("%Y-%m-%d 00:00:00"),
                                today.strftime("%Y-%m-%d 23:59:59")))

        condition.time_periods = time_periods

        # 解析置信度
        conf_patterns = [
            (r'置信度\s*(?:大于|超过|高于|>=)\s*(\d+(?:\.\d+)?)\s*%?', 'min'),
            (r'置信度\s*(?:小于|低于|<=)\s*(\d+(?:\.\d+)?)\s*%?', 'max'),
            (r'准确率\s*(?:大于|超过|高于|>=)\s*(\d+(?:\.\d+)?)\s*%?', 'min'),
        ]
        for pattern, conf_type in conf_patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = float(match.group(1))
                if value <= 100:
                    value = value / 100
                if conf_type == 'min':
                    condition.confidence_min = value
                else:
                    condition.confidence_max = value
                break

        # 解析摄像头
        if cameras:
            for cam in cameras:
                cam_name = cam.get('name', '')
                cam_id = cam.get('id', '')
                if cam_name and cam_name in query:
                    condition.camera_id = cam_id
                    condition.camera_name = cam_name
                    break

        # 解析数量限制
        limit_match = re.search(r'(?:前|最新|最近)\s*(\d+)\s*(?:条|个|次)', query)
        if limit_match:
            condition.limit = min(int(limit_match.group(1)), 100)

        return condition

    def _llm_parse(self, query: str, cameras: List[Dict] = None) -> QueryCondition:
        """
        使用 LLM 解析自然语言查询
        """
        camera_info = ""
        if cameras:
            camera_info = "可用摄像头列表：\n" + "\n".join([
                f"- ID: {cam['id']}, 名称: {cam['name']}"
                for cam in cameras
            ])

        today = datetime.now().strftime("%Y-%m-%d")

        system_prompt = f"""将人脸识别查询转为JSON。当前日期:{today}
{camera_info}

输出格式(仅JSON):
{{"person_names":["姓名"],"time_periods":[["开始时间","结束时间"]],"confidence_min":0.9,"camera_id":"id","limit":50}}

规则:
- person_names: 姓名数组，多人用数组，单人如["张三"]
- time_periods: 时间段数组，格式YYYY-MM-DD HH:MM:SS
- 置信度百分比转小数(90%->0.9)
- 未提及字段省略
- 仅输出JSON"""

        user_prompt = f"请解析以下查询：{query}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
        except requests.exceptions.Timeout:
            raise Exception("LLM API 调用超时，请检查网络连接")
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM API 网络错误: {e}")

        if response.status_code != 200:
            raise Exception(f"LLM API 调用失败: {response.status_code}")

        result = response.json()
        content = result['choices'][0]['message']['content']

        # 解析 LLM 返回的 JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(content)

        condition = QueryCondition()

        # 解析人员姓名列表
        if data.get('person_names'):
            names = data['person_names']
            if isinstance(names, str):
                names = [names]
            condition.person_names = [n for n in names if n]

        # 解析时间段列表
        if data.get('time_periods'):
            periods = data['time_periods']
            for period in periods:
                if isinstance(period, list) and len(period) >= 2:
                    condition.time_periods.append((period[0], period[1]))

        # 兼容旧格式（单个 person_name 和 start_time/end_time）
        if data.get('person_name') and not condition.person_names:
            condition.person_names = [data['person_name']]
        if data.get('start_time') and data.get('end_time') and not condition.time_periods:
            condition.time_periods.append((data['start_time'], data['end_time']))

        if data.get('confidence_min') is not None:
            condition.confidence_min = float(data['confidence_min'])
        if data.get('confidence_max') is not None:
            condition.confidence_max = float(data['confidence_max'])
        if data.get('camera_id'):
            condition.camera_id = data['camera_id']
        if data.get('limit'):
            condition.limit = min(int(data['limit']), 100)

        return condition

    def build_sql_query(self, condition: QueryCondition) -> tuple:
        """
        根据查询条件构建 SQL 查询
        支持多人员（OR）和多时间段（OR）查询

        Returns:
            (sql, params) 元组
        """
        sql = "SELECT id, person_name, confidence, timestamp, camera_id, camera_name FROM recognition_history WHERE 1=1"
        params = []

        # 多人员查询（OR 条件）
        if condition.person_names:
            if len(condition.person_names) == 1:
                sql += " AND person_name LIKE ?"
                params.append(f"%{condition.person_names[0]}%")
            else:
                placeholders = " OR ".join(["person_name LIKE ?" for _ in condition.person_names])
                sql += f" AND ({placeholders})"
                for name in condition.person_names:
                    params.append(f"%{name}%")

        if condition.confidence_min is not None:
            sql += " AND confidence >= ?"
            params.append(condition.confidence_min)

        if condition.confidence_max is not None:
            sql += " AND confidence <= ?"
            params.append(condition.confidence_max)

        # 多时间段查询（OR 条件）
        if condition.time_periods:
            if len(condition.time_periods) == 1:
                sql += " AND timestamp >= ? AND timestamp <= ?"
                params.extend([condition.time_periods[0][0], condition.time_periods[0][1]])
            else:
                time_conditions = []
                for start, end in condition.time_periods:
                    time_conditions.append("(timestamp >= ? AND timestamp <= ?)")
                    params.extend([start, end])
                sql += f" AND ({' OR '.join(time_conditions)})"

        if condition.camera_id:
            sql += " AND camera_id = ?"
            params.append(condition.camera_id)

        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([condition.limit, condition.offset])

        return sql, params


# 全局实例
_llm_service = None

def get_llm_service() -> LLMService:
    """获取 LLM 服务单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
