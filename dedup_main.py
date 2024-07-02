import json
import traceback
from typing import Dict, List

from client.api import Client
from helper.constant import DEDUP_LOCK_EXPIRE_TIME
from helper.dedup_es_utils import DedupESUtils
from helper.dedup_verify_utils import DedupVerifyUtils
from helper.education import EducationHelper
from helper.logger import getLogger, redact_profile
from helper.position import PositionHelper
from helper.redis_lock_utils import RedisLock
from helper.social_links import clean_linkedin_url, clean_social_urls
from helper.utility import UtilityHelper
from service import merge_profile
from service.matcher import DebugProfileMatcher, MatchResult, ProfileMatcher


class DiscoverService:
    def __init__(self):
        self.logger = getLogger()
        self.es_utils = DedupESUtils()
        self.verify_utils = DedupVerifyUtils()
        self.redis_lock = RedisLock()
        self.merge_profile = merge_profile.MergeProfileService()

    # for api call, need merge is false, for consumer consume, need merge is true
    def find_dup_by_id(self, team_id, talent_id, check_email=False, need_merge=False, debug=False):
        if not talent_id:
            return None
        if not team_id:
            team_id = "htmtalent"
        log_prefix = f"[find_dup_by_id] {team_id}||{talent_id}"
        original_profile = Client.get_profiles(team_id, [talent_id])
        if not original_profile or not original_profile.get(talent_id):
            self.logger.error(f"{log_prefix} no profile")
            return None
        original_profile = original_profile.get(talent_id)
        # if not original_profile.get('position') and not original_profile.get('education'):
        #     self.logger.error(f'{team_id}||{talent_id} profile not rich enough')
        #     return None
        result = self.find_dup_by_profile(team_id, original_profile, check_email, need_merge, debug=debug)
        self.logger.info(
            f"{log_prefix} find dup result %s",
            json.dumps(result, ensure_ascii=False),
        )
        return result

    # only dup discover consumer has need_merge default as true
    # only search existing uid has relax_strictness default as true
    def find_dup_by_profile(
        self, team_id, original_profile, check_email=False, need_merge=False, relax_strictness=False, debug=False
    ):
        # map to store lock_key and lok
        key_lock_map = None
        if not original_profile:
            return {}
        if not team_id:
            team_id = "htmtalent"
        # when need to merge, be strict
        if need_merge:
            relax_strictness = False
        talent_id = ""
        basic = original_profile.get("basic", {})
        if not basic:
            self.logger.error(
                f"[find_dup_by_profile] {team_id} original profile have no basic {redact_profile(original_profile)}"
            )
            return {}
        try:
            basic = clean_social_urls(basic)
            original_profile["basic"] = basic
        except Exception as e:
            self.logger.error(f"[find_dup_by_profile] clean_social_urls failed, {basic} {e}")
        # Get talent id.
        if basic.get("user_id"):
            talent_id = original_profile["basic"]["user_id"]
        log_prefix = f"[find_dup_by_profile] {team_id}||{talent_id}"
        try:
            self.logger.info(
                f"{log_prefix} check_email: {check_email}, need_merge: {need_merge}, relax_strictness: {relax_strictness}, debug: {debug}, original_profile: {json.dumps(redact_profile(original_profile), ensure_ascii=False)}"
            )
            verified_id_list = []
            suspected_id_list = []
            unverified_id_list = []
            manual_verified_id_list, manual_suspected_id_list, manual_unverified_id_list = [], [], []
            debug_manual_verified_id_list, debug_manual_suspected_id_list, debug_manual_unverified_id_list = [], [], []

            all_redirect_linkedin_url = set()
            target_linkedin_url = None
            talent_id_linkedin_url_map = {}

            # if linkedin url is present, need to check if there is redirect url, then use all linkedin urls to do cross-ref
            if basic.get("linkedin_url"):
                try:
                    original_linkedin_url = clean_linkedin_url(basic["linkedin_url"])
                    if original_linkedin_url:
                        if talent_id:
                            talent_id_linkedin_url_map[talent_id] = original_linkedin_url
                        res_map = Client.get_redirect_linkedin_url(original_linkedin_url)
                        if res_map.get("status") and res_map.get("redirect_urls") and res_map.get("target_url"):
                            all_redirect_linkedin_url = set(res_map.get("redirect_urls"))
                            target_linkedin_url = res_map.get("target_url")
                            self.logger.info(
                                "%s: all redirect linkedin_url for %s: %s, target url: %s",
                                log_prefix,
                                original_linkedin_url,
                                all_redirect_linkedin_url,
                                target_linkedin_url,
                            )
                        else:
                            all_redirect_linkedin_url = {original_linkedin_url}
                            target_linkedin_url = original_linkedin_url
                        for linkedin_url in all_redirect_linkedin_url:
                            linkedin_url = clean_linkedin_url(linkedin_url)
                            if linkedin_url:
                                uid = Client.get_id_by_social({"linkedin_url": linkedin_url})
                                if uid and uid != talent_id:
                                    talent_id_linkedin_url_map[uid] = linkedin_url
                                    verified_id_list.append(uid)
                                    self.logger.info("%s find by linkedin_url %s: %s", log_prefix, linkedin_url, uid)
                except Exception:
                    self.logger.error(f'{log_prefix} find by linkedin_url failed, {basic.get("linkedin_url")}')

            similar_uids = []
            id_similar_title_map = {}
            # es find similar uid based on name, email, social, pos and edu
            try:
                self.get_profile_pos_edu_ids(original_profile, log_prefix)
                similar_uids = self.search_similar_uids(team_id, original_profile, log_prefix, check_email)
                self.logger.info("%s es result similar uids: %s", log_prefix, similar_uids)
            except Exception:
                self.logger.error("%s error searching similar uids, error: %s", log_prefix, traceback.format_exc())

            # remove uid from similar uids if it's already verified
            similar_uids = [similar_uid for similar_uid in similar_uids if similar_uid not in verified_id_list]
            # lock all related uids
            keys_to_lock = list(dict.fromkeys(similar_uids + verified_id_list))
            if talent_id:
                keys_to_lock.append(talent_id)
            key_lock_map = self.redis_lock.multilock(keys_to_lock, DEDUP_LOCK_EXPIRE_TIME)
            self.logger.info(f"{log_prefix} acquired locks for {list(key_lock_map.keys())}")
            dup_profile_map = dict()
            if similar_uids:
                try:
                    dup_profile_map = Client.get_profiles(
                        team_id, similar_uids + [talent_id] if talent_id else similar_uids
                    )
                    if not talent_id:
                        dup_profile_map[talent_id] = original_profile
                    elif not dup_profile_map.get(talent_id):
                        raise Exception(f"{log_prefix} no profiles for target id")
                    verified_by_linkedin_list = []
                    for similar_uid in similar_uids:
                        similar_profile = dup_profile_map.get(similar_uid, {})
                        if not similar_profile:
                            dup_profile_map.pop(similar_uid, {})
                            continue

                        similar_basic = similar_profile.get("basic", {})
                        if not similar_basic:
                            continue
                        similar_linkedin = clean_linkedin_url(similar_basic.get("linkedin_url"))
                        # verified if linkedin in redirect linkedin urls
                        if similar_linkedin:
                            talent_id_linkedin_url_map[similar_uid] = similar_linkedin
                            if all_redirect_linkedin_url and similar_linkedin in all_redirect_linkedin_url:
                                verified_by_linkedin_list.append(similar_uid)
                                self.logger.info(
                                    "%s find by redirect linkedin_url %s, %s", log_prefix, similar_uid, similar_linkedin
                                )
                    # remove uid from similar uids if it's already verified
                    if verified_by_linkedin_list:
                        verified_id_list = list(dict.fromkeys(verified_id_list + verified_by_linkedin_list))
                        similar_uids = [
                            similar_uid for similar_uid in similar_uids if similar_uid not in verified_id_list
                        ]
                        dup_profile_map = {
                            similar_uid: similar_profile
                            for similar_uid, similar_profile in dup_profile_map.items()
                            if similar_uid not in verified_id_list
                        }
                    if not dup_profile_map:
                        if similar_uids:
                            raise Exception(
                                f"{log_prefix} no profiles for duplicate talent: {json.dumps(similar_uids)}"
                            )
                        else:
                            raise Exception(f"{log_prefix} all similar uids verified by linkedin")

                    # get similar title map from ml
                    for id in dup_profile_map.keys():
                        id_similar_title_map[id] = {}
                    original_positions = original_profile.get("position", [])
                    for id, profile in dup_profile_map.items():
                        if id == talent_id:
                            continue
                        id_similar_title_map[talent_id][id] = PositionHelper.get_similar_title_map(
                            original_positions, profile.get("position", []), f"{log_prefix}||{id}"
                        )
                        id_similar_title_map[id][talent_id] = PositionHelper.reverse_title_map(
                            id_similar_title_map.get(talent_id, {}).get(id, {})
                        )

                    # verify all similar uids with target id
                    (
                        debug_manual_verified_id_list,
                        debug_manual_suspected_id_list,
                        debug_manual_unverified_id_list,
                    ) = self.manual_check_similar_id_debug(
                        dup_profile_map,
                        talent_id,
                        similar_uids,
                        id_similar_title_map,
                        log_prefix,
                        relax_strictness,
                        debug=debug,
                    )
                    (
                        manual_verified_id_list,
                        manual_suspected_id_list,
                        manual_unverified_id_list,
                    ) = self.manual_check_similar_id(
                        dup_profile_map, talent_id, similar_uids, id_similar_title_map, log_prefix, relax_strictness
                    )
                    for verified_id in manual_verified_id_list:
                        self.logger.info(
                            f"{log_prefix} manual check found verified {verified_id}:{redact_profile(dup_profile_map.get(verified_id))}"
                        )
                except Exception:
                    e = traceback.format_exc()
                    self.logger.error(f"{log_prefix} manual check exception: {e}")

            # add manual check verified/suspect result into result list
            # Be aware, order is important here, manual check have sorted the verified id list by match score desc, should maintain that order after array merge
            verified_id_list = list(dict.fromkeys(verified_id_list + manual_verified_id_list))
            suspected_id_list = list(
                set().union(suspected_id_list, manual_suspected_id_list).difference(verified_id_list)
            )
            # check whether default profile is also duplicate, since merge happens on profile level, not team level
            # even though team id is htmtalent, need to verify here, because there might be multiple verified ids, better verify each pair of them before merge them.
            if relax_strictness:
                verified_tuple = [verified_id_list, suspected_id_list]
            elif debug and not need_merge:
                verified_tuple = [debug_manual_verified_id_list, debug_manual_suspected_id_list]
            else:
                verified_tuple = self.verify_by_default_profile(
                    verified_id_list, suspected_id_list, talent_id, id_similar_title_map
                )
                # if after verified by default profile, the result changed, log it down
                if len(verified_id_list) != len(verified_tuple[0]):
                    self.logger.info(
                        "%s verified has difference before(%s) after(%s)",
                        log_prefix,
                        json.dumps(verified_id_list, ensure_ascii=False),
                        json.dumps(verified_tuple[0], ensure_ascii=False),
                    )
                if len(suspected_id_list) != len(verified_tuple[1]):
                    self.logger.info(
                        "%s suspected has difference before(%s) after(%s)",
                        log_prefix,
                        json.dumps(suspected_id_list, ensure_ascii=False),
                        json.dumps(verified_tuple[1], ensure_ascii=False),
                    )
            unverified_id_list = list(set(similar_uids).difference(verified_tuple[0], verified_tuple[1]))
            result = {
                "verified": verified_tuple[0],
                "suspected": verified_tuple[1],
                "unverified": unverified_id_list,
            }

            debug_same = set(debug_manual_verified_id_list) == set(manual_verified_id_list) and set(
                debug_manual_suspected_id_list
            ) == set(manual_suspected_id_list)
            # log manual check result
            debug_compare_res = dict()
            if similar_uids:
                debug_compare_res = {
                    "team_id": team_id,
                    "target_id": talent_id,
                    "manual": {"verified": manual_verified_id_list, "suspected": manual_suspected_id_list},
                    "debug": {"verified": debug_manual_verified_id_list, "suspected": debug_manual_suspected_id_list},
                    "debug_same": debug_same,
                    "target_profile": redact_profile(original_profile),
                    "similar_uids": similar_uids,
                }
                self.logger.info("%s data to verify: %s", log_prefix, json.dumps(debug_compare_res, ensure_ascii=False))

            # merge verified candidates
            if need_merge and result["verified"]:
                # partially unlock key that's not verified or not the target, i.e candidate that's not going to merge
                all_lock_key_list = list(key_lock_map.keys())
                if talent_id and talent_id in all_lock_key_list:
                    UtilityHelper.safe_remove_from_list(all_lock_key_list, talent_id)
                for verified_id in result["verified"]:
                    UtilityHelper.safe_remove_from_list(all_lock_key_list, verified_id)
                if all_lock_key_list:
                    self.redis_lock.unlock_partial_multilock(key_lock_map, all_lock_key_list)
                    self.logger.info(
                        f"{log_prefix} partially unlock {all_lock_key_list}, remaining locks: {list(key_lock_map.keys())}"
                    )

                verified_id_list = list(result["verified"])
                try:
                    # save all linkedin urls into a map
                    linkedin_url_id_set = set()
                    for candidate_id in [*verified_id_list, talent_id]:
                        linkedin_url = talent_id_linkedin_url_map.get(candidate_id)
                        # only leave target linkedin and other linkedin url that's not redirect url of target linkedin
                        if linkedin_url and (
                            linkedin_url == target_linkedin_url or linkedin_url not in all_redirect_linkedin_url
                        ):
                            linkedin_url_id_set.add(linkedin_url)
                    # need to save linkedin url map when there are multiple linkedin url among verified candidates
                    if len(linkedin_url_id_set) > 1:
                        linkedin_url_list = list(linkedin_url_id_set)
                        save_res = Client.save_richest_linkedin_url_map(linkedin_url_list)
                        self.logger.info(f"{log_prefix} save linkedin url map {linkedin_url_list}, res: {save_res}")

                    # merge candidates
                    res = Client.merge_duplicate_profiles(talent_id, verified_id_list)
                    if res and res.get("targetId"):
                        self.logger.info(f"{log_prefix} merged {verified_id_list} => {talent_id}, {res}")
                        # put the target id to the front of verified list.
                        if not talent_id:
                            index_of_res = result["verified"].index(res.get("targetId"))
                            result["verified"][index_of_res] = result["verified"][0]
                            result["verified"][0] = res.get("targetId")
                        talent_id = res.get("targetId")
                        try:
                            self.merge_profile.merge_duplicate_profiles(talent_id, verified_id_list, dup_profile_map)
                        except Exception as e:
                            self.logger.exception(
                                f"{log_prefix} merge_duplicate_profiles failed {verified_id_list} => {talent_id}, {e}"
                            )
                    else:
                        self.logger.error(f"{log_prefix} merge failed {verified_id_list} => {talent_id}, {res}")
                except Exception:
                    e = traceback.format_exc()
                    self.logger.error(f"{log_prefix} merge exception, error: {e}")
        finally:
            # always unlock all remaining keys in the end
            if key_lock_map:
                self.redis_lock.unlock_multilock(key_lock_map.values())
                self.logger.info(f"{log_prefix} unlock remaining locks: {list(key_lock_map.keys())}")
        return result

    def search_similar_uids(self, team_id, original_profile, log_prefix, check_email=False):
        try:
            # shorten original_profile string if possible
            try:
                basic = original_profile.get("basic") or {}
                user_id = basic.get("user_id")
                if user_id:
                    original_profile_string = f"original_profile_id({str(user_id)})"
                else:
                    profile_string = json.dumps(redact_profile(original_profile), ensure_ascii=False)
                    if len(profile_string) < 8000:
                        original_profile_string = f"original_profile({profile_string})"
                    else:
                        basic_string = json.dumps(basic, ensure_ascii=False)
                        original_profile_string = f"original_profile_basic({basic_string})"
            except Exception:
                original_profile_string = json.dumps(redact_profile(original_profile), ensure_ascii=False)

            es_query = self.es_utils.generate_es_query(team_id, original_profile, check_email)
            if not es_query:
                raise Exception(f"{log_prefix} no es query {original_profile_string}")
            tid2score = Client.es_query_talents(es_query)
            if not tid2score:
                # raise Exception(f'no es result {team_id}||{original_profile}')
                similar_uids = []
            else:
                similar_uids = list(tid2score.keys())
            return similar_uids
        except Exception:
            e = traceback.format_exc()
            self.logger.error(f"{log_prefix} es exception: {e}")
            return []

    def get_profile_pos_edu_ids(self, profile, log_prefix):
        if not profile:
            return
        pos = profile.get("position", [])
        edu = profile.get("education", [])
        if pos:
            try:
                position_list = []
                for position in pos:
                    if (position.get("position_company_name") or position.get("norm_domain")) and not position.get(
                        "company_id"
                    ):
                        position_list.append(position)
                if position_list:
                    company_ids = PositionHelper.get_company_ids_from_position_list(position_list, log_prefix)
                    for index, position in enumerate(position_list):
                        position["company_id"] = UtilityHelper.safe_get_list(company_ids, index, "")
            except Exception:
                e = traceback.format_exc()
                self.logger.error(f"{log_prefix} failed to get pos id for {redact_profile(profile)}", e)
        if edu:
            try:
                education_list = []
                for education in edu:
                    if education.get("education_school") and not education.get("education_id"):
                        education_list.append(education)
                if education_list:
                    education_ids = EducationHelper.get_school_ids_from_education_list(education_list, log_prefix)
                    for index, education in enumerate(education_list):
                        education_id = UtilityHelper.safe_get_list(education_ids, index, "")
                        if education_id:
                            education["education_id"] = [education_id]
                        else:
                            education["education_id"] = []
            except Exception:
                e = traceback.format_exc()
                self.logger.error(f"{log_prefix} failed to get edu id for {redact_profile(profile)}", e)

    def search_existing_id_by_profile(self, team_id, profile, need_merge=False):
        if not profile:
            return None
        try:
            self.logger.info(
                "[search_existing_id_by_profile] start %s %s",
                team_id,
                json.dumps(redact_profile(profile), ensure_ascii=False),
            )
            if profile.get("basic"):
                profile["basic"]["user_id"] = ""
            # default not merge and relax strictness, since it's mainly used for import
            result = self.find_dup_by_profile(team_id, profile, True, need_merge, True)
            self.logger.info(
                "[search_existing_id_by_profile] end %s %s: %s",
                team_id,
                json.dumps(redact_profile(profile), ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )
            if not result or not result.get("verified"):
                return None
            verified_ids = result.get("verified")
            return verified_ids[0]
        except Exception:
            e = traceback.format_exc()
            self.logger.error(e)
            return None

    def manual_check_similar_id_debug(
        self,
        profile_map,
        original_id,
        similar_uids,
        id_similar_title_map,
        log_prefix,
        relax_strictness=False,
        debug=False,
    ):
        verified_list = []
        verified_list_with_score = []
        suspected_list = []
        unverified_list = []
        original_profile = profile_map.get(original_id, {})
        if not original_profile:
            self.logger.error(f"{log_prefix} manual check debug no original profile")
            return verified_list, suspected_list, unverified_list
        original_basic = original_profile.get("basic", {})
        if not original_basic:
            return verified_list, suspected_list, unverified_list
        for similar_uid in similar_uids:
            if similar_uid == original_id:
                continue
            inner_log_prefix = f"{log_prefix}||{similar_uid} manual check debug(relax:{relax_strictness})"
            similar_profile = profile_map.get(similar_uid)
            if not similar_profile:
                self.logger.error(f"{inner_log_prefix} no similar profile")
                continue
            self.logger.info(f"{inner_log_prefix} similar profile: {redact_profile(similar_profile)}")
            try:
                similar_basic = similar_profile.get("basic", {})
                if not similar_basic:
                    continue
                matcher = DebugProfileMatcher(
                    original_profile,
                    similar_profile,
                    id_similar_title_map.get(original_id, {}).get(similar_uid, {}),
                    inner_log_prefix,
                    relax_strictness,
                    debug,
                )
                match_result, score = matcher.match()
                if match_result == MatchResult.VERIFIED:
                    if score is None:
                        verified_list.append(similar_uid)
                    else:
                        verified_list_with_score.append((similar_uid, score))
                elif match_result == MatchResult.SUSPECTED:
                    suspected_list.append(similar_uid)
                else:
                    unverified_list.append(similar_uid)
            except Exception as e:
                self.logger.error(f"{inner_log_prefix} ERROR: {e}")
        # sort all verified by match score
        if verified_list_with_score:
            sorted_verified_list = sorted(verified_list_with_score, key=lambda item: item[1], reverse=True)
            verified_list += [item[0] for item in sorted_verified_list]
        return verified_list, suspected_list, unverified_list

    def manual_check_similar_id(
        self, profile_map, original_id, similar_uids, id_similar_title_map, log_prefix, relax_strictness=False
    ):
        verified_list = []
        verified_list_with_score = []
        suspected_list = []
        unverified_list = []
        original_profile = profile_map.get(original_id, {})
        if not original_profile:
            self.logger.error(f"{log_prefix} manual check no original profile")
            return verified_list, suspected_list, unverified_list
        original_basic = original_profile.get("basic", {})
        if not original_basic:
            return verified_list, suspected_list, unverified_list
        for similar_uid in similar_uids:
            if similar_uid == original_id:
                continue
            inner_log_prefix = f"{log_prefix}||{similar_uid} manual check(relax:{relax_strictness})"
            similar_profile = profile_map.get(similar_uid)
            if not similar_profile:
                self.logger.error(f"{inner_log_prefix} no similar profile")
                continue
            self.logger.info(f"{inner_log_prefix} similar profile: {redact_profile(similar_profile)}")
            try:
                similar_basic = similar_profile.get("basic", {})
                if not similar_basic:
                    continue
                matcher = ProfileMatcher(
                    original_profile,
                    similar_profile,
                    id_similar_title_map.get(original_id, {}).get(similar_uid, {}),
                    inner_log_prefix,
                    relax_strictness,
                )
                match_result, score = matcher.match()
                if match_result == MatchResult.VERIFIED:
                    if score is None:
                        verified_list.append(similar_uid)
                    else:
                        verified_list_with_score.append((similar_uid, score))
                elif match_result == MatchResult.SUSPECTED:
                    suspected_list.append(similar_uid)
                else:
                    unverified_list.append(similar_uid)
            except Exception as e:
                self.logger.error(f"{inner_log_prefix} ERROR: {e}")
        # sort all verified by match score
        if verified_list_with_score:
            sorted_verified_list = sorted(verified_list_with_score, key=lambda item: item[1], reverse=True)
            verified_list += [item[0] for item in sorted_verified_list]
        return verified_list, suspected_list, unverified_list

    def verify_by_default_profile(self, verified_id_list, suspected_id_list, talent_id, id_similar_title_map):
        all_uids = list(set(verified_id_list + suspected_id_list))
        if talent_id and talent_id not in all_uids:
            all_uids.append(talent_id)
        default_profile_map = Client.get_profiles("htmtalent", all_uids)
        verified_ids_without_default_profile = verified_id_list
        suspected_ids_without_default_profile = suspected_id_list
        if default_profile_map:
            verified_ids_without_default_profile = [
                uid for uid in verified_id_list if self.verify_utils.profile_lack_info(default_profile_map.get(uid))
            ]
            suspected_ids_without_default_profile = [
                uid for uid in suspected_id_list if self.verify_utils.profile_lack_info(default_profile_map.get(uid))
            ]
        original_default_profile = default_profile_map.get(talent_id, None) if talent_id else None
        verified_groups = []
        log_prefix = f"[verify_by_default_profile] htmtalent||{talent_id}"
        for verified_id in verified_id_list:
            if verified_id in verified_ids_without_default_profile:
                continue
            if not verified_groups:
                verified_groups.append([verified_id])
                continue
            else:
                for i, group in enumerate(verified_groups):
                    verified_list, suspected_list, unverified_list = self.manual_check_similar_id(
                        default_profile_map, verified_id, group, id_similar_title_map, log_prefix
                    )
                    if unverified_list:
                        verified_groups.append([verified_id])
                        break
                    else:
                        verified_groups[i].append(verified_id)
                        break
        if not verified_groups:
            verified = verified_ids_without_default_profile
        elif not original_default_profile:
            verified = list(set(verified_groups[0] + verified_ids_without_default_profile))
        else:
            verified_group = []
            for group in verified_groups:
                verified_list, suspected_list, unverified_list = self.manual_check_similar_id(
                    default_profile_map, talent_id, group, id_similar_title_map, log_prefix
                )
                if verified_list:
                    verified_group = group
                    break
                if verified_group:
                    break
            verified = verified_group

        suspcted_groups = []
        for suspected_id in suspected_id_list:
            if suspected_id in suspected_ids_without_default_profile:
                continue
            if not suspcted_groups:
                suspcted_groups.append([suspected_id])
                continue
            else:
                for i, group in enumerate(suspcted_groups):
                    verified_list, suspected_list, unverified_list = self.manual_check_similar_id(
                        default_profile_map, suspected_id, group, id_similar_title_map, log_prefix
                    )
                    if unverified_list:
                        suspcted_groups.append([suspected_id])
                        break
                    else:
                        suspcted_groups[i].append(suspected_id)
                        break
        if not suspcted_groups:
            suspected = suspected_ids_without_default_profile
        elif not original_default_profile:
            suspected = list(set(suspcted_groups[0] + suspected_ids_without_default_profile))
        else:
            suspcted_group = []
            for group in suspcted_groups:
                verified_list, suspected_list, unverified_list = self.manual_check_similar_id(
                    default_profile_map, talent_id, group, id_similar_title_map, log_prefix
                )
                if suspected_list or verified_list:
                    suspcted_group = group
                    break
                if suspcted_group:
                    break
            suspected = suspcted_group
        return verified, suspected

    def merge_duplicate_profiles_from_api(
        self, target_id: str, source_id_list: List[str], group_id: str, team_id: str, target_profile: Dict
    ) -> bool:
        key_lock_map = self.redis_lock.multilock([target_id] + source_id_list, DEDUP_LOCK_EXPIRE_TIME)
        self.logger.info(f"Acquired locks: {list(key_lock_map.keys())}")
        try:
            self.merge_profile.merge_duplicate_profiles_from_api(
                target_id, source_id_list, group_id, team_id, target_profile
            )
        except Exception:
            self.logger.exception(f"Failed to merge duplicate profiles from api: {target_id} {source_id_list}")
            return False
        finally:
            if key_lock_map:
                self.redis_lock.unlock_multilock(key_lock_map.values())
                self.logger.info(f"Release locks: {list(key_lock_map.keys())}")
        return True
