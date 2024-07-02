from enum import Enum
from dataclasses import dataclass, field, asdict

from helper.logger import getLogger
from helper.constant import SOCIAL_LINKS, SOCIAL_VALUE_BLACKLIST
from helper.dedup_feature_utils import generate_name_list, check_linkedin_url, generate_name_list_new
from helper.position import PositionHelper
from helper.education import EducationHelper

from typing import Optional, Tuple, Union

logger = getLogger()


class MatchResult(str, Enum):
    VERIFIED = "verified"
    SUSPECTED = "suspected"
    UNVERIFIED = "unverified"

    def __str__(self):
        return self.value


@dataclass
class MatchContext:
    name_match: Union[int, float] = -1
    social_match: int = -1
    position_match: Union[int, float] = -1
    education_match: Union[int, float] = -1

    # social_same_count: int = 0

    min_pos_count: int = 0
    # position_match_count: int = 0
    pos_exact_match: int = 0

    min_edu_count: int = 0

    details: dict = field(default_factory=dict)
    match_result: Optional[MatchResult] = None
    match_score: Union[int, float, None] = None

    @property
    def is_end(self):
        return self.match_result is not None


class ProfileMatcher:
    def __init__(
        self,
        original_profile: dict,
        similar_profile: dict,
        similar_title_map: dict,
        log_prefix: str,
        relax_strictness: bool = False,
    ):
        self.original_profile = original_profile
        self.similar_profile = similar_profile
        self.similar_title_map = similar_title_map
        self.log_prefix = log_prefix
        self.relax_strictness = relax_strictness
        self.context = MatchContext()

    def set_result(self, result: MatchResult, socre: Union[int, float, None] = None):
        self.context.match_result = result
        self.context.match_score = socre

    def _update_detail(self, key: str, key_detail: dict):
        self.context.details[key] = key_detail

    def match_by_name(self):
        original_fullname = self.original_profile.get("basic", {}).get("fullname") or ""
        similar_fullname = self.similar_profile.get("basic", {}).get("fullname") or ""
        # verify name match
        original_name_list = generate_name_list(original_fullname)
        similar_name_list = generate_name_list(similar_fullname)

        if not original_name_list or not similar_name_list:
            self.context.name_match = 0
        elif set(original_name_list) & set(similar_name_list):
            self.context.name_match = 1
        elif original_fullname in similar_fullname or similar_fullname in original_fullname:
            self.context.name_match = 0

        ctx = {
            "original": {"fullname": original_fullname, "name_list": original_name_list},
            "similar": {"fullname": similar_fullname, "name_list": similar_name_list},
        }
        self._update_detail("name_match", ctx)

    def match_by_socials(self):
        """Depends on name_match"""
        original_basic = self.original_profile.get("basic", {})
        similar_basic = self.similar_profile.get("basic", {})

        # verify social match
        same_socials = []
        diff_socials = []
        verified_by_linkedin = False
        for url_key in SOCIAL_LINKS:
            if not original_basic.get(url_key) or not similar_basic.get(url_key):
                continue
            original_social = original_basic[url_key].strip().lower()
            similar_social = similar_basic[url_key].strip().lower()
            if original_social in SOCIAL_VALUE_BLACKLIST or similar_social in SOCIAL_VALUE_BLACKLIST:
                continue
            if url_key == "linkedin_url":
                if check_linkedin_url(original_social, similar_social, self.context.name_match):
                    verified_by_linkedin = True
                    break
            if original_social == similar_social:
                same_socials.append(url_key)
            else:
                diff_socials.append(url_key)
        social_same_count = len(same_socials)
        social_dif_count = len(diff_socials)

        # verify email match
        original_emails = set().union(
            original_basic.get("default_emails") or [], original_basic.get("edited_email") or []
        )
        similar_emails = set().union(similar_basic.get("default_emails") or [], similar_basic.get("edited_email") or [])
        if original_emails & similar_emails:
            same_socials.append("emails")
            social_same_count += 1

        self.context.social_match = 1 if social_same_count > 0 else 0
        # if there are different socials, don't take socials into account when matching
        if social_dif_count > 0:
            self.context.social_match = 0

        ctx = {
            "verified_by_linkedin": verified_by_linkedin,
            "same_socials": same_socials,
            "social_same_count": social_same_count,
            "diff_socials": diff_socials,
            "social_dif_count": social_dif_count,
        }
        self._update_detail("social_match", ctx)

        if verified_by_linkedin:
            self.set_result(MatchResult.VERIFIED)
            logger.info(f"{self.log_prefix}: verified by linkedin")

    def match_by_positions(self):
        # verify position match
        positions_1 = self.original_profile.get("position") or []
        positions_2 = self.similar_profile.get("position") or []
        self.context.min_pos_count = 0
        if not positions_1 or not positions_2:
            self.context.position_match = 0
        else:
            similar_title_map = self.similar_title_map
            if len(positions_1) > len(positions_2):
                positions_1, positions_2 = positions_2, positions_1
                similar_title_map = PositionHelper.reverse_title_map(similar_title_map)
            self.context.min_pos_count = len(positions_1)
            position_match_count = 0
            matched_p2_index = set()
            rough_matched_p2_index = set()
            for _, p1 in enumerate(positions_1):
                exact_match = False
                rough_match = False
                temp_rough_match = set()
                for p2_index, p2 in enumerate(positions_2):
                    if p2_index in matched_p2_index:
                        continue
                    company_same = PositionHelper.is_same_company(p1, p2)
                    title_same = PositionHelper.is_same_title(p1, p2, 20, similar_title_map)
                    start_date_same = PositionHelper.is_start_date_same(p1, p2, 12)
                    end_date_same = PositionHelper.is_end_date_same(p1, p2, 12)
                    if company_same == -1:
                        continue
                    # exact match: company and title and one of date match
                    if company_same == 1 and title_same == 1:
                        if start_date_same == 1 or end_date_same == 1:
                            exact_match = True
                            matched_p2_index.add(p2_index)
                            break
                    # exact match: start date match and end date not mismatch, company match or title match(company not mismatch)
                    if start_date_same == 1 and (end_date_same == 1 or end_date_same == 0):
                        if company_same == 1 or company_same == 0 and title_same == 1:
                            exact_match = True
                            matched_p2_index.add(p2_index)
                            break
                    if p2_index in rough_matched_p2_index:
                        continue
                    # rough match: company or title match, start date or end date match
                    if (company_same == 1 or title_same == 1) and (start_date_same == 1 or end_date_same == 1):
                        rough_match = True
                        temp_rough_match.add(p2_index)
                        continue
                    # rough match: company and title, start date and end date not mismatch
                    if company_same == 1 and title_same == 1 and start_date_same == 0 and end_date_same == 0:
                        rough_match = True
                        temp_rough_match.add(p2_index)
                        continue
                    # rough match: when relax, any one of factors match and no mismatch
                    if (
                        self.relax_strictness
                        and company_same + title_same + start_date_same + end_date_same >= 1
                        and company_same != -1
                        and title_same != -1
                        and start_date_same != -1
                        and end_date_same != -1
                    ):
                        rough_match = True
                        temp_rough_match.add(p2_index)
                        continue
                if exact_match:
                    position_match_count += 2
                elif rough_match:
                    position_match_count += 1
                    rough_matched_p2_index.add(temp_rough_match.pop())

            # overall exact match condition is if more than half of the pos is exact match or multiple rough match and at least one exact match
            if position_match_count > len(positions_1) or (
                self.relax_strictness and position_match_count >= len(positions_1)
            ):
                self.context.position_match = 1
            # overall rough match condition is if more than half of the pos is rough match
            elif position_match_count * 2 > len(positions_1):
                self.context.position_match = 0
            self._update_detail(
                "position_match",
                {"position_match_count": position_match_count, "min_pos_count": self.context.min_pos_count},
            )

    def match_by_educations(self):
        # verify education match
        educations_1 = self.original_profile.get("education") or []
        educations_2 = self.similar_profile.get("education") or []
        self.context.min_edu_count = 0
        if not educations_1 or not educations_2:
            self.context.education_match = 0
        else:
            if len(educations_1) > len(educations_2):
                educations_1, educations_2 = educations_2, educations_1
            self.context.min_edu_count = len(educations_1)
            educations_1 = EducationHelper.format_edu(educations_1)
            educations_2 = EducationHelper.format_edu(educations_2)
            matched_e2_index = set()
            rough_matched_e2_index = set()
            education_match_count = 0
            for _, e1 in enumerate(educations_1):
                exact_match = False
                rough_match = False
                temp_rough_match = set()
                for e2_index, e2 in enumerate(educations_2):
                    if e2_index in matched_e2_index:
                        continue
                    school_same = EducationHelper.is_same_school(e1, e2)
                    degree_same = EducationHelper.is_same_degree(e1, e2)
                    major_same = EducationHelper.is_same_major(e1, e2)
                    year_same = EducationHelper.is_same_year(e1, e2)
                    # exact match: school match, any two of degree, major, year match
                    if school_same == 1:
                        if (
                            year_same == 1
                            and major_same == 1
                            and degree_same != -1
                            or year_same == 1
                            and degree_same == 1
                            and major_same != -1
                            or major_same == 1
                            and degree_same == 1
                            and year_same != -1
                        ):
                            exact_match = True
                            matched_e2_index.add(e2_index)
                            break
                    if e2_index in rough_matched_e2_index:
                        continue
                    # rough match: none of four edu factors is mismatch
                    if (
                        school_same != -1
                        and year_same != -1
                        and major_same != -1
                        and degree_same != -1
                        and school_same + year_same + major_same + degree_same >= 1
                    ):
                        rough_match = True
                        temp_rough_match.add(e2_index)
                        continue
                if exact_match:
                    education_match_count += 2
                elif rough_match:
                    education_match_count += 1
                    rough_matched_e2_index.add(temp_rough_match.pop())

            # overall exact match condition is if more than half of the edu is exact match or multiple rough match and at least one exact match
            # since one often has one or two edu, the condition in most cases is at least one exact match, and the others must at least be rough match
            if education_match_count > len(educations_1):
                self.context.education_match = 1
            # overall rough match condition is at least all edu are rough match
            elif education_match_count == len(educations_1):
                self.context.education_match = 0
            self._update_detail(
                "education_match",
                {"education_match_count": education_match_count, "min_edu_count": self.context.min_edu_count},
            )

    def match_by_context(self):
        # total match score
        ctx = self.context
        total_match = ctx.social_match + ctx.position_match + ctx.education_match + ctx.name_match
        # relax strictness
        if self.relax_strictness:
            # can be verify when name and one of factor among three other is match
            if ctx.name_match == 1 and (ctx.social_match == 1 or ctx.position_match == 1 or ctx.education_match == 1):
                self.set_result(MatchResult.VERIFIED, total_match)
                return
            # can be verify when name not match, social and pos/edu match
            elif ctx.social_match == 1 and (
                ctx.position_match == 1
                or ctx.education_match == 1
                or (ctx.name_match >= 0 and ctx.min_pos_count == 0 and ctx.min_edu_count == 0)
            ):
                self.set_result(MatchResult.VERIFIED, total_match)
                return
        # if one of the profile lack pos/edu info, verify only when name, pos, edu match
        if not self.relax_strictness and ctx.min_pos_count <= 1 and ctx.min_edu_count <= 1:
            # if both pos and edu empty, can verify by name and socials
            if ctx.min_pos_count == 0 and ctx.min_edu_count == 0:
                if ctx.name_match == 1 and ctx.social_match == 1:
                    self.set_result(MatchResult.VERIFIED, total_match)
                elif ctx.social_match == 1:
                    self.set_result(MatchResult.SUSPECTED, total_match)
                else:
                    self.set_result(MatchResult.UNVERIFIED, total_match)
            elif (ctx.min_pos_count == 0 or ctx.position_match == 1) and (
                ctx.min_edu_count == 0 or ctx.education_match == 1
            ):
                if ctx.name_match == 1:
                    self.set_result(MatchResult.VERIFIED, total_match)
                else:
                    self.set_result(MatchResult.SUSPECTED, total_match)
            else:
                self.set_result(MatchResult.UNVERIFIED, total_match)
        # four factors considered for matching: name, social, position, education
        # verify condition
        # 1. when two of them is exact match(1) and the other two is rough match(0)
        # 2. when three of them is exact match(1) and the other one is mismatch(-1) or rough match(0)
        # 3. when all four of them is exact match(1)
        elif (
            ctx.social_match + ctx.position_match + ctx.education_match + ctx.name_match >= 2
            and ctx.position_match != -1
            and ctx.education_match != -1
        ):
            self.set_result(MatchResult.VERIFIED, total_match)
        # suspect condition
        # 1. at least one of social/pos/edu is exact match and other is not mismatch
        # 2. two exact match and one mismatch
        elif ctx.social_match + ctx.position_match + ctx.education_match >= 1:
            self.set_result(MatchResult.SUSPECTED, total_match)
        else:
            self.set_result(MatchResult.UNVERIFIED, total_match)

    def match(self) -> Tuple[Optional[MatchResult], Union[int, float, None]]:
        try:
            for method in [
                self.match_by_name,
                self.match_by_socials,
                self.match_by_positions,
                self.match_by_educations,
                self.match_by_context,
            ]:
                method()
                if self.context.is_end:
                    return self.context.match_result, self.context.match_score
            if not self.context.is_end:
                logger.warning(f"{self.log_prefix} null match_result, context: {asdict(self.context)}")
                raise ValueError("Match result is None, please check")
            return self.context.match_result, self.context.match_score
        except Exception as e:
            logger.exception(f"{self.log_prefix} match error: {e}")
            raise e
        finally:
            logger.info(f"{self.log_prefix} context: {asdict(self.context)}")


class DebugProfileMatcher(ProfileMatcher):
    def __init__(
        self,
        original_profile: dict,
        similar_profile: dict,
        similar_title_map: dict,
        log_prefix: str,
        relax_strictness: bool = False,
        debug: bool = False,
    ):
        self.original_profile = original_profile
        self.similar_profile = similar_profile
        self.similar_title_map = similar_title_map
        self.log_prefix = log_prefix
        self.relax_strictness = relax_strictness
        self.debug = debug
        self.context = MatchContext()

    def match_by_name(self):
        original_fullname = self.original_profile.get("basic", {}).get("fullname") or ""
        similar_fullname = self.similar_profile.get("basic", {}).get("fullname") or ""
        # verify name match
        original_name_list, original_short_name_list, original_name_part_list = generate_name_list_new(
            original_fullname
        )
        similar_name_list, similar_short_name_list, similar_name_part_list = generate_name_list_new(similar_fullname)
        if not original_name_list or not similar_name_list:
            self.context.name_match = 0
        elif set(original_name_list) & set(similar_name_list):
            self.context.name_match = 1
        elif set(original_name_list) & set(similar_short_name_list) or set(original_short_name_list) & set(
            similar_name_list
        ):
            self.context.name_match = 0.5
        elif (
            original_fullname in similar_fullname
            or similar_fullname in original_fullname
            or set(original_name_part_list) & set(similar_name_part_list)
        ):
            self.context.name_match = 0

        ctx = {
            "original": {
                "fullname": original_fullname,
                "name_list": original_name_list,
                "short_name_list": original_short_name_list,
                "name_part_list": original_name_part_list,
            },
            "similar": {
                "fullname": similar_fullname,
                "name_list": similar_name_list,
                "short_name_list": similar_short_name_list,
                "name_part_list": similar_name_part_list,
            },
        }
        self._update_detail("name_match", ctx)

    def match_by_positions(self):
        # verify position match
        positions_1 = self.original_profile.get("position", [])
        positions_2 = self.similar_profile.get("position", [])
        self.context.min_pos_count = 0
        self.context.pos_exact_match = 0
        if not positions_1 or not positions_2:
            self.context.position_match = 0
        else:
            # only compare the experience that have overlapping time range
            similar_title_map = self.similar_title_map
            positions_1, positions_2 = PositionHelper.filter_comparable_positions(positions_1, positions_2)
            if len(positions_1) > len(positions_2):
                positions_1, positions_2 = positions_2, positions_1
                similar_title_map = PositionHelper.reverse_title_map(similar_title_map)
            # if debug:
            #     company_similarity = Client.get_company_similarity([positions_1, positions_2])
            #     company_id_similarity = company_similarity.get("company_id_similarity", [])
            #     company_str_similarity = company_similarity.get("name_string_similarity", [])
            # else:
            #     company_id_similarity, company_str_similarity = None, None
            company_id_similarity, company_str_similarity = [], []
            self.context.min_pos_count = len(positions_1)
            position_match_count = 0
            matched_p2_index = set()
            rough_matched_p2_index = set()
            for p1_index, p1 in enumerate(positions_1):
                exact_match = False
                rough_match = False
                temp_rough_match = set()
                for p2_index, p2 in enumerate(positions_2):
                    if p2_index in matched_p2_index:
                        continue
                    if not company_id_similarity and not company_str_similarity:
                        company_same = PositionHelper.is_same_company(p1, p2)
                    else:
                        cid_same = company_id_similarity[p1_index][p2_index]
                        str_same = company_str_similarity[p1_index][p2_index]
                        company_same = max(cid_same, str_same)
                    title_same = PositionHelper.is_same_title(p1, p2, 20, similar_title_map)
                    start_date_same = PositionHelper.is_start_date_same(p1, p2, 12)
                    end_date_same = PositionHelper.is_end_date_same(p1, p2, 12)
                    if company_same == -1:
                        continue
                    # exact match: company and title and one of date match
                    if company_same == 1 and title_same == 1:
                        if start_date_same == 1 or end_date_same == 1:
                            exact_match = True
                            matched_p2_index.add(p2_index)
                            break
                    # exact match: start date match and end date not mismatch, company match or title match(company not mismatch)
                    if start_date_same == 1 and (end_date_same == 1 or end_date_same == 0):
                        if company_same == 1 or company_same == 0 and title_same == 1:
                            exact_match = True
                            matched_p2_index.add(p2_index)
                            break
                    if p2_index in rough_matched_p2_index:
                        continue
                    if company_same == 1 and title_same == 1:
                        rough_match = True
                        temp_rough_match.add(p2_index)
                        continue
                    # rough match: company or title match, start date or end date match
                    if (company_same == 1 or title_same == 1) and (start_date_same == 1 or end_date_same == 1):
                        rough_match = True
                        temp_rough_match.add(p2_index)
                        continue
                    # rough match: company and title, start date and end date not mismatch
                    if company_same == 1 and title_same == 1 and start_date_same == 0 and end_date_same == 0:
                        rough_match = True
                        temp_rough_match.add(p2_index)
                        continue
                    # rough match: when relax, any one of factors match and no mismatch
                    if (
                        self.relax_strictness
                        and company_same + title_same + start_date_same + end_date_same >= 1
                        and company_same != -1
                        and title_same != -1
                        and start_date_same != -1
                        and end_date_same != -1
                    ):
                        rough_match = True
                        temp_rough_match.add(p2_index)
                        continue
                if exact_match:
                    position_match_count += 2
                    self.context.pos_exact_match += 1
                elif rough_match:
                    position_match_count += 1
                    rough_matched_p2_index.add(temp_rough_match.pop())

            # overall exact match condition is if more than half of the pos is exact match or multiple rough match and at least one exact match
            if position_match_count > len(positions_1) or (
                self.relax_strictness and position_match_count >= len(positions_1)
            ):
                self.context.position_match = 1
            elif (
                self.context.min_pos_count <= 1
                and position_match_count > 0
                and position_match_count >= len(positions_1)
            ):
                self.context.position_match = 0.5
            # overall rough match condition is if more than half of the pos is rough match
            elif position_match_count * 2 > len(positions_1) or not positions_1 or not positions_2:
                self.context.position_match = 0
            self._update_detail(
                "position_match",
                {
                    "position_match_count": position_match_count,
                    "min_pos_count": self.context.min_pos_count,
                    "pos_exact_match": self.context.pos_exact_match,
                },
            )

    def match_by_educations(self):
        # verify education match
        educations_1 = self.original_profile.get("education", [])
        educations_2 = self.similar_profile.get("education", [])
        self.context.min_edu_count = 0
        if not educations_1 or not educations_2:
            self.context.education_match = 0
        else:
            if len(educations_1) > len(educations_2):
                educations_1, educations_2 = educations_2, educations_1
            self.context.min_edu_count = len(educations_1)
            educations_1 = EducationHelper.format_edu(educations_1)
            educations_2 = EducationHelper.format_edu(educations_2)
            matched_e2_index = set()
            rough_matched_e2_index = set()
            education_match_count = 0
            for _, e1 in enumerate(educations_1):
                exact_match = False
                rough_match = False
                temp_rough_match = set()
                for e2_index, e2 in enumerate(educations_2):
                    if e2_index in matched_e2_index:
                        continue
                    school_same = EducationHelper.is_same_school(e1, e2)
                    degree_same = EducationHelper.is_same_degree(e1, e2)
                    major_same = EducationHelper.is_same_major(e1, e2)
                    year_same = EducationHelper.is_same_year(e1, e2)
                    # exact match: school match, any two of degree, major, year match
                    if school_same == 1:
                        if (
                            year_same == 1
                            and major_same == 1
                            and degree_same != -1
                            or year_same == 1
                            and degree_same == 1
                            and major_same != -1
                            or major_same == 1
                            and degree_same == 1
                            and year_same != -1
                        ):
                            exact_match = True
                            matched_e2_index.add(e2_index)
                            break
                    if e2_index in rough_matched_e2_index:
                        continue
                    # rough match: none of four edu factors is mismatch
                    if (
                        school_same != -1
                        and year_same != -1
                        and major_same != -1
                        and degree_same != -1
                        and school_same + year_same + major_same + degree_same >= 1
                    ):
                        rough_match = True
                        temp_rough_match.add(e2_index)
                        continue
                if exact_match:
                    education_match_count += 2
                elif rough_match:
                    education_match_count += 1
                    rough_matched_e2_index.add(temp_rough_match.pop())

            # overall exact match condition is if more than half of the edu is exact match or multiple rough match and at least one exact match
            # since one often has one or two edu, the condition in most cases is at least one exact match, and the others must at least be rough match
            if education_match_count > len(educations_1) or (
                self.relax_strictness and education_match_count > 0 and education_match_count >= len(educations_1)
            ):
                self.context.education_match = 1
            elif (
                self.context.min_edu_count <= 1
                and education_match_count > 0
                and education_match_count >= len(educations_1)
            ):
                self.context.education_match = 0.5
            # overall rough match condition is at least all edu are rough match
            elif education_match_count == len(educations_1):
                self.context.education_match = 0
            self._update_detail(
                "education_match",
                {"education_match_count": education_match_count, "min_edu_count": self.context.min_edu_count},
            )

    def match_by_context(self):
        ctx = self.context
        is_verified = False
        is_suspected = False
        # total match score
        total_match = ctx.social_match + ctx.position_match + ctx.education_match + ctx.name_match
        # relax strictness
        if self.relax_strictness:
            # can be verify when name and one of factor among three other is match
            if ctx.name_match == 1 and (ctx.social_match == 1 or ctx.position_match == 1 or ctx.education_match == 1):
                is_verified = True
            # can be verify when name not match, social and pos/edu match
            if ctx.social_match == 1 and (ctx.position_match == 1 or ctx.education_match == 1):
                is_verified = True
        # if one of the profile lack pos/edu info, verify only when name, pos, edu match
        if ctx.min_pos_count <= 1 or ctx.min_edu_count <= 1:
            # if both pos and edu empty, can verify by name and socials
            if ctx.min_pos_count == 0 and ctx.min_edu_count == 0:
                if ctx.name_match == 1 and ctx.social_match == 1:
                    is_verified = True
                if ctx.social_match == 1:
                    is_suspected = True
            elif (ctx.min_pos_count == 0 or ctx.position_match > 0) and (
                ctx.min_edu_count == 0 or ctx.education_match > 0
            ):
                if ctx.name_match == 1:
                    is_verified = True
                else:
                    is_suspected = True
        # four factors considered for matching: name, social, position, education
        # verify condition
        # 1. when two of them is exact match(1) and the other two is rough match(0)
        # 2. when three of them is exact match(1) and the other one is mismatch(-1) or rough match(0)
        # 3. when all four of them is exact match(1)
        if (
            ctx.social_match + ctx.position_match + ctx.education_match + ctx.name_match >= 2
            and ctx.position_match != -1
            and ctx.education_match != -1
        ):
            is_verified = True
        # Jul6 2023 add logic: half above position are exactly match are highly suspected to be same person
        if ctx.pos_exact_match > 2 and ctx.position_match:
            if ctx.social_match > 0 or ctx.name_match > 0 or ctx.education_match > 0:
                is_verified = True
            else:
                is_suspected = True
        # suspect condition
        # 1. at least one of social/pos/edu is exact match and other is not mismatch
        # 2. two exact match and one mismatch
        if ctx.social_match + ctx.position_match + ctx.education_match >= 1:
            is_suspected = True

        if is_verified:
            self.set_result(MatchResult.VERIFIED, total_match)
        elif is_suspected:
            self.set_result(MatchResult.SUSPECTED)
        else:
            self.set_result(MatchResult.UNVERIFIED)
