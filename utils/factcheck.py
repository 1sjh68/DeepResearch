# utils/factcheck.py

import logging
import re
from dataclasses import dataclass
from datetime import datetime

# NLTK停用词库
try:
    import nltk  # noqa: F401
    from nltk.corpus import stopwords
    # 尝试加载停用词，如果失败则下载
    try:
        stop_words_en = set(stopwords.words('english'))
        stop_words_zh = set(stopwords.words('chinese'))
        NLTK_AVAILABLE = True
    except LookupError:
        # 停用词数据未下载，标记为不可用
        NLTK_AVAILABLE = False
        logging.info("NLTK stopwords not downloaded. Use nltk.download('stopwords') to enable.")
except ImportError:
    NLTK_AVAILABLE = False
    logging.debug("nltk not installed. Using fallback stopwords list.")


@dataclass
class FactCheckResult:
    """事实核查结果"""

    claim: str
    is_verifiable: bool
    verification_score: float
    supporting_sources: list[str]
    contradicting_sources: list[str]
    confidence_level: str  # "high", "medium", "low", "unverifiable"
    notes: str = ""
    last_updated: str = ""


@dataclass
class VerificationRule:
    """验证规则"""

    pattern: str
    rule_type: str  # "numeric", "temporal", "authoritative", "cross_reference"
    min_sources: int = 2
    confidence_threshold: float = 0.7


class FactChecker:
    """轻量事实核查工具"""

    def __init__(self):
        self.verification_rules = self._load_verification_rules()
        self.source_reliability = {
            "academic": 0.9,
            "government": 0.9,
            "news": 0.7,
            "blog": 0.4,
            "social": 0.2,
            "unknown": 0.5,
        }

    def _load_verification_rules(self) -> list[VerificationRule]:
        """加载验证规则"""
        return [
            # 数值验证规则
            VerificationRule(
                pattern=r"(\d+(?:\.\d+)?)\s*(%|percent)",
                rule_type="numeric",
                min_sources=2,
                confidence_threshold=0.8,
            ),
            # 时间验证规则
            VerificationRule(
                pattern=r"(19|20)\d{2}",
                rule_type="temporal",
                min_sources=1,
                confidence_threshold=0.6,
            ),
            # 权威性声明验证
            VerificationRule(
                pattern=r"(研究表明|调查显示|根据.*研究|权威.*报告)",
                rule_type="authoritative",
                min_sources=2,
                confidence_threshold=0.7,
            ),
            # 交叉引用验证
            VerificationRule(
                pattern=r"(与其他.*一致|与.*相符|与.*一致)",
                rule_type="cross_reference",
                min_sources=3,
                confidence_threshold=0.8,
            ),
        ]

    def check_minimal(self, claims: list[str], sources: list[dict]) -> list[FactCheckResult]:
        """执行最小事实核查"""
        results = []

        for claim in claims:
            result = self._check_single_claim(claim, sources)
            results.append(result)

        logging.info(f"完成 {len(claims)} 个主张的事实核查")
        return results

    def _check_single_claim(self, claim: str, sources: list[dict]) -> FactCheckResult:
        """检查单个主张"""
        supporting_sources = []
        contradicting_sources = []
        verification_score = 0.0
        confidence_level = "unverifiable"

        # 1. 检查主张是否可验证
        is_verifiable = self._is_verifiable_claim(claim)

        if not is_verifiable:
            return FactCheckResult(
                claim=claim,
                is_verifiable=False,
                verification_score=0.0,
                supporting_sources=[],
                contradicting_sources=[],
                confidence_level="unverifiable",
                notes="主张包含个人观点或无法验证的内容",
                last_updated=datetime.now().isoformat(),
            )

        # 2. 在源中搜索支持证据
        supporting_sources, contradicting_sources = self._search_evidence(claim, sources)

        # 3. 计算验证分数
        verification_score = self._calculate_verification_score(claim, supporting_sources, contradicting_sources)

        # 4. 确定置信度等级
        confidence_level = self._determine_confidence_level(verification_score, len(supporting_sources))

        # 5. 生成说明
        notes = self._generate_verification_notes(claim, supporting_sources, contradicting_sources, verification_score)

        return FactCheckResult(
            claim=claim,
            is_verifiable=is_verifiable,
            verification_score=verification_score,
            supporting_sources=[s.get("title", s.get("url", "")) for s in supporting_sources],
            contradicting_sources=[s.get("title", s.get("url", "")) for s in contradicting_sources],
            confidence_level=confidence_level,
            notes=notes,
            last_updated=datetime.now().isoformat(),
        )

    def _is_verifiable_claim(self, claim: str) -> bool:
        """判断主张是否可验证"""
        # 不可验证的内容模式
        unverifiable_patterns = [
            r"我认为",
            r"我觉得",
            r"个人观点",
            r"可能.*(?=认为|觉得|估计|猜测)",
            r"也许|或许|大概|估计|猜测",
            r"未来.*会",
            r"将来.*可能",
        ]

        for pattern in unverifiable_patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                return False

        # 可验证的内容特征
        verifiable_indicators = [
            r"\d{4}",  # 年份
            r"\d+(?:\.\d+)?\s*%",  # 百分比
            r"根据.*研究",
            r"数据显示",
            r"统计.*显示",
            r"调查.*发现",
        ]

        for indicator in verifiable_indicators:
            if re.search(indicator, claim):
                return True

        # 如果包含具体名词和数据，认为可验证
        if len(claim.split()) > 5 and re.search(r"\d|研究|调查|数据", claim):
            return True

        return False

    def _search_evidence(self, claim: str, sources: list[dict]) -> tuple[list[dict], list[dict]]:
        """在源中搜索支持或矛盾证据"""
        supporting_sources = []
        contradicting_sources = []

        # 提取关键词
        keywords = self._extract_keywords(claim)

        for source in sources:
            content = source.get("summary", "") + " " + source.get("content", "")
            relevance_score = self._calculate_relevance(claim, content, keywords)

            if relevance_score > 0.3:  # 相关性阈值
                # 检查支持还是矛盾
                support_type = self._classify_support_type(claim, content)

                if support_type == "support":
                    source_with_score = source.copy()
                    source_with_score["relevance_score"] = relevance_score
                    supporting_sources.append(source_with_score)
                elif support_type == "contradict":
                    source_with_score = source.copy()
                    source_with_score["relevance_score"] = relevance_score
                    contradicting_sources.append(source_with_score)

        # 按相关性排序
        supporting_sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        contradicting_sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return supporting_sources, contradicting_sources

    def _extract_keywords(self, claim: str) -> list[str]:
        """从主张中提取关键词（使用NLTK停用词库）"""
        # 使用NLTK停用词库（如果可用）
        if NLTK_AVAILABLE:
            stop_words = stop_words_en | stop_words_zh
        else:
            # 回退到基本停用词列表
            stop_words = {
                "的", "了", "在", "是", "有", "和", "与", "及", "或", "但", "而", "则",
                "the", "is", "are", "and", "or", "but", "in", "on", "at", "to", "of",
                "a", "an", "as", "by", "for", "from", "it", "that", "this", "with",
            }

        # 分词并过滤
        words = re.findall(r"\b\w+\b", claim.lower())
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # 优先保留数字和专业术语
        priority_words = [word for word in keywords if re.search(r"\d|研究|调查|统计", word)]

        return priority_words or keywords[:5]  # 最多返回5个关键词

    def _calculate_relevance(self, claim: str, content: str, keywords: list[str]) -> float:
        """计算内容相关性"""
        if not keywords or not content:
            return 0.0

        content_lower = content.lower()
        claim_lower = claim.lower()

        # 关键词匹配分数
        keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
        keyword_score = keyword_matches / len(keywords)

        # 语义相似度分数（简单版本）
        semantic_score = 0.0
        claim_words = set(claim_lower.split())
        content_words = set(content_lower.split())

        if claim_words and content_words:
            overlap = len(claim_words.intersection(content_words))
            semantic_score = overlap / len(claim_words.union(content_words))

        # 综合分数
        final_score = keyword_score * 0.7 + semantic_score * 0.3
        return min(1.0, final_score)

    def _classify_support_type(self, claim: str, content: str) -> str:
        """分类支持类型：支持、矛盾或中立"""
        content_lower = content.lower()

        # 支持性词汇
        support_indicators = [
            "证实",
            "确认",
            "支持",
            "符合",
            "一致",
            "显示",
            "表明",
            "confirm",
            "support",
            "consistent",
            "show",
            "indicate",
        ]

        # 矛盾性词汇
        contradict_indicators = [
            "矛盾",
            "冲突",
            "不符",
            "反对",
            "否定",
            "质疑",
            "挑战",
            "contradict",
            "conflict",
            "oppose",
            "deny",
            "question",
            "challenge",
        ]

        support_score = sum(1 for indicator in support_indicators if indicator in content_lower)
        contradict_score = sum(1 for indicator in contradict_indicators if indicator in content_lower)

        if contradict_score > support_score:
            return "contradict"
        elif support_score > contradict_score:
            return "support"
        else:
            # 检查数值一致性
            claim_numbers = re.findall(r"\d+(?:\.\d+)?", claim)
            content_numbers = re.findall(r"\d+(?:\.\d+)?", content)

            if claim_numbers and content_numbers:
                # 简单的数值一致性检查
                for num in claim_numbers:
                    if num in content:
                        return "support"

            return "neutral"  # 中立

    def _calculate_verification_score(self, claim: str, supporting: list[dict], contradicting: list[dict]) -> float:
        """计算验证分数"""
        base_score = 0.0

        # 基础分数：支持源数量
        support_bonus = min(len(supporting) * 0.3, 0.6)
        base_score += support_bonus

        # 惩罚分数：矛盾源数量
        contradict_penalty = min(len(contradicting) * 0.4, 0.4)
        base_score -= contradict_penalty

        # 规则匹配加分
        rule_bonus = self._calculate_rule_bonus(claim)
        base_score += rule_bonus

        # 源可靠性加权
        if supporting:
            reliability_score = sum(self._get_source_reliability(s) for s in supporting) / len(supporting)
            base_score *= reliability_score

        return max(0.0, min(1.0, base_score))

    def _calculate_rule_bonus(self, claim: str) -> float:
        """根据验证规则计算加分"""
        bonus = 0.0

        for rule in self.verification_rules:
            if re.search(rule.pattern, claim):
                if rule.rule_type == "numeric":
                    bonus += 0.2
                elif rule.rule_type == "temporal":
                    bonus += 0.1
                elif rule.rule_type == "authoritative":
                    bonus += 0.3
                elif rule.rule_type == "cross_reference":
                    bonus += 0.25

        return min(0.5, bonus)  # 最大加分限制

    def _get_source_reliability(self, source: dict) -> float:
        """获取源可靠性分数"""
        url = source.get("url", "").lower()
        title = source.get("title", "").lower()
        content = source.get("content", "").lower()

        # 基于域名和内容特征判断可靠性
        if any(domain in url for domain in [".edu", ".gov", ".org"]):
            return self.source_reliability["academic"]
        elif any(domain in url for domain in ["reuters", "bbc", "cnn", "ap.org"]):
            return self.source_reliability["news"]
        elif any(indicator in title + " " + content for indicator in ["研究", "调查", "统计", "report", "study"]):
            return self.source_reliability["academic"]
        else:
            return self.source_reliability["unknown"]

    def _determine_confidence_level(self, verification_score: float, source_count: int) -> str:
        """确定置信度等级"""
        if verification_score >= 0.8 and source_count >= 2:
            return "high"
        elif verification_score >= 0.6 and source_count >= 1:
            return "medium"
        elif verification_score >= 0.3:
            return "low"
        else:
            return "unverifiable"

    def _generate_verification_notes(self, claim: str, supporting: list[dict], contradicting: list[dict], score: float) -> str:
        """生成验证说明"""
        notes = []

        if not supporting and not contradicting:
            notes.append("未找到相关证据来源")
        else:
            if supporting:
                notes.append(f"找到 {len(supporting)} 个支持证据")
            if contradicting:
                notes.append(f"发现 {len(contradicting)} 个矛盾证据")

        if score >= 0.8:
            notes.append("高置信度：主张得到充分支持")
        elif score >= 0.6:
            notes.append("中等置信度：主张得到部分支持")
        elif score >= 0.3:
            notes.append("低置信度：主张证据不足")
        else:
            notes.append("无法验证：主张缺乏支持证据")

        return "；".join(notes)

    def generate_unverifiable_report(self, fact_check_results: list[FactCheckResult]) -> dict:
        """生成不可验证主张报告"""
        unverifiable = [r for r in fact_check_results if not r.is_verifiable or r.confidence_level == "unverifiable"]
        low_confidence = [r for r in fact_check_results if r.confidence_level in ["low"] and r.is_verifiable]

        report = {
            "total_claims": len(fact_check_results),
            "unverifiable_claims": len(unverifiable),
            "low_confidence_claims": len(low_confidence),
            "verification_rate": (len(fact_check_results) - len(unverifiable)) / len(fact_check_results) if fact_check_results else 0,
            "unverifiable_list": [
                {
                    "claim": result.claim,
                    "reason": result.notes,
                    "confidence": result.confidence_level,
                }
                for result in unverifiable
            ],
            "low_confidence_list": [
                {
                    "claim": result.claim,
                    "verification_score": result.verification_score,
                    "supporting_sources": len(result.supporting_sources),
                    "notes": result.notes,
                }
                for result in low_confidence
            ],
            "recommendations": self._generate_recommendations(unverifiable, low_confidence),
            "generated_at": datetime.now().isoformat(),
        }

        return report

    def _generate_recommendations(self, unverifiable: list[FactCheckResult], low_confidence: list[FactCheckResult]) -> list[str]:
        """生成改进建议"""
        recommendations = []

        if unverifiable:
            recommendations.append(f"需要为 {len(unverifiable)} 个不可验证主张寻找更多可靠证据")

        if low_confidence:
            recommendations.append(f"需要补充 {len(low_confidence)} 个低置信度主张的支持证据")

        if not recommendations:
            recommendations.append("所有主张都有充分的证据支持")

        # 基于具体问题给出建议
        if any("个人观点" in result.notes for result in unverifiable):
            recommendations.append("建议将个人意见与客观事实分离")

        if any(result.verification_score < 0.3 for result in low_confidence):
            recommendations.append("建议寻找更多权威来源来验证争议性主张")

        return recommendations


def add_verification_markers(text: str, fact_check_results: list[FactCheckResult]) -> str:
    """在文本中添加验证标记"""
    marked_text = text

    # 收集需要标记的结果及其位置
    markers_to_insert: list[tuple[int, str]] = []

    for result in fact_check_results:
        if result.confidence_level in ["low", "unverifiable"]:
            # 找到主张在原始文本中的位置
            claim_start = text.find(result.claim)
            if claim_start >= 0:
                claim_end = claim_start + len(result.claim)

                # 根据置信度选择标记
                if result.confidence_level == "unverifiable":
                    marker = " <!-- 无法验证 -->"
                else:
                    marker = f" <!-- 验证分数: {result.verification_score:.2f} -->"

                markers_to_insert.append((claim_end, marker))

    # 按位置从后往前排序，这样前面的插入不会影响后面的位置索引
    markers_to_insert.sort(key=lambda x: x[0], reverse=True)

    # 从后往前插入标记
    for claim_end, marker in markers_to_insert:
        marked_text = marked_text[:claim_end] + marker + marked_text[claim_end:]

    return marked_text
