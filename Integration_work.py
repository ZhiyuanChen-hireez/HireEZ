from openai import AsyncOpenAI
import asyncio
import numpy as np
import json
from itertools import combinations, chain
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
import time
import asyncio

api_key = open("api_key_unlimited.txt").read().strip()

class ProfileSimilarityCalculator:
    def __init__(self, api_key, model="text-embedding-3-large"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.DEGREE_ALIAS = {
            "highschool": ["high_school", "high school", "highschool", "high-school", "secondary school", "secondaryschool", "secondary-school"],
            "associate": ["associate", "associates", "associate's", "associates'", "associate bachelor", "associate bs"],
            "bs": ["bachelor", "bachelors", "bachelor's", "bachelors'", "undergraduate", "undergrad", "undergrads", "undergrad's", "undergrads'", "bs", "ba", "b.sc", "b.sc.", "bsc", "bachelor of science", "bachelor of arts", "b.a.","b.a"],
            "ma": ["master", "masters", "master's", "masters'", "graduate", "grad", "grads", "grad's", "grads'", "ms", "ma", "m.sc", "m.sc.", "msc", "master of science", "master of arts", "m.a.","m.a"],
            "phd": ["phd", "ph.d", "doctorate", "doctoral", "doctor", "doctor's", "doctorate of philosophy", "doctorate of philosophy", "doctor of philosophy"],
            "md": ["md", "m.d", "doctor of medicine", "doctorate of medicine"],
            "do" : ["do", "d.o", "doctor of osteopathic medicine", "doctorate of osteopathic medicine"],
            "jd": ["jd", "j.d", "juris doctor", "juris doctorate"],
            "pharmd": ["pharmd", "pharm.d", "doctor of pharmacy", "doctorate of pharmacy"],
            "mba": ["mba", "m.b.a", "master of business administration"],
            "dba": ["dba", "d.b.a", "doctor of business administration"],
        }

    async def get_embedding(self, text, max_retries=5, initial_delay=1, timeout=20):
        start = time.time()

        if not text:
            return []

        if type(text) != list:
            text = [text]

        retries = 0
        delay = initial_delay
        while retries < max_retries:
            task = asyncio.create_task(self.client.embeddings.create(input=text, model=self.model))
            try:
                response = await asyncio.wait_for(task, timeout=timeout)
                end = time.time()
                print(f"Start time {start}, end time {end}, Time taken: {end-start}")
                return [r.embedding for r in response.data]
            except asyncio.TimeoutError:
                print(f"Attempt {retries + 1} failed: Timeout")
            except Exception as e:
                print(f"Attempt {retries + 1} failed: {e}")
            retries += 1
            if retries >= max_retries:
                print(f"Failed after {max_retries} attempts.")
                raise Exception("Failed after {max_retries} attempts.")
            wait_time = delay * (2 ** retries)
            print(f"Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
            start = time.time()

        end = time.time()
        print(f"Start time {start}, end time {end}, Time taken: {end-start}")
        return []

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def flatten_values(values):
        for value in values:
            if isinstance(value, list):
                for item in value:
                    yield item
            else:
                yield value

    def string_from_education(self, education):
        string = ""
        if not education:
            return "null"
        for i in education:
            i["education_id"] = i["education_school_linkedin_url"] = i['education_school_logo'] = ''
            string += " ".join(value for value in ProfileSimilarityCalculator.flatten_values(i.values()))
            string += "."
        return string

    def strings_from_education(self, education):
        strings = []
        for i in education:
            i["education_id"] = i["education_school_linkedin_url"] = i['education_school_logo'] = ''
            strings.append(" ".join(value for value in ProfileSimilarityCalculator.flatten_values(i.values())))
        return strings

    def degree_normalization(self, degree):
        if not degree:
            return " "
        
        degree = degree.lower()
        for key, values in self.DEGREE_ALIAS.items():
            if any(value in degree for value in values):
                return key

        return degree

# helper function to calculate the similarity between two education entries
    def calc_school_similarity(self, target, similar):
        if target['education_school'] == '' or similar['education_school'] == '':
            return 1
        return np.dot(target['education_school_embedding'], similar['education_school_embedding'])

    def calc_degree_similarity(self, target, similar):
        return np.dot(target['education_degree_embedding'], similar['education_degree_embedding'])

    def calc_major_similarity(self, target, similar):
        if target['education_major'] in similar['education_major'] or similar['education_major'] in target['education_major']:
            return 1

        index = 0
        if target['education_major'] != '' and similar['education_major'] != '':
            index += 4
        if target['education_description'] != '' and similar['education_description'] != '':
            index += 2
        if target['education_activities'] != '' and similar['education_activities'] != '':
            index += 1
        
        target_embedding = target['education_major_embedding'][index]
        similar_embedding = similar['education_major_embedding'][index]

        return np.dot(target_embedding, similar_embedding)

    def calc_time_similarity(self, target, similar):
        # initialize time similarity to 1 (and as for no all time stamps available)
        time_similarity = 1
        if target['education_startyear'] == '' or similar['education_startyear'] == '':
            if target['education_endyear'] == '' or similar['education_endyear'] == '':
                return 1
            return int(target['education_endyear'] == similar['education_endyear'])
        elif target['education_endyear'] == '' or similar['education_endyear'] == '':
            return int(target['education_startyear'] == similar['education_startyear'])

        # at this point we know that both startyear and endyear are available
        target_startyear = int(target['education_startyear'])
        target_endyear = int(target['education_endyear'])
        similar_startyear = int(similar['education_startyear'])
        similar_endyear = int(similar['education_endyear'])
        value = max(0, min(target_endyear, similar_endyear) - max(target_startyear, similar_startyear))
        target_duration = abs(target_endyear - target_startyear)
        similar_duration = abs(similar_endyear - similar_startyear)
        if target_duration == 0 and similar_duration == 0:
            time_similarity = int(target_startyear == similar_startyear)
        else:
            time_similarity = 2 * value / (target_duration + similar_duration)
        return time_similarity

    # return a list of dictionaries of information (check variable result) for each education entry
    async def get_weighted_embedding(self, education):
        async def process_single_education(e):
            result = {
                'education_embedding': 0, 
                'education_school_embedding': 0, 
                'education_degree_embedding': 0, 
                'education_major_embedding': [], 
                'education_year_score': 0, 
                'education_string': '', 
                'education_school': '', 
                'education_degree': '', 
                'education_major': '', 
                'education_description': '',
                'education_activities': '',
                'education_startyear': '', 
                'education_endyear': ''
            }
            result['education_school'] = e['education_school'] if e['education_school'] != "" else " "
            result['education_degree'] = e['education_degree_level'] if e['education_degree_level'] != "" else self.degree_normalization(e['education_degree'])
            result['education_major'] = e['education_major'] if e['education_major'] != "" else " "
            result['education_description'] = e['education_description'] if e['education_description'] != "" else " "
            result['education_activities'] = e['education_activities'] if e['education_activities'] != "" else " "
            result['education_startyear'] = e['education_startyear']
            result['education_endyear'] = e['education_endyear'] 
            result['education_string'] = self.string_from_education([e])

            embeddings = await (
                self.get_embedding([
                    result['education_school'], 
                    result['education_degree'],
                    result['education_school'] + " " + result['education_degree'] + " " + result['education_major'],
                    result['education_major'],
                    result['education_major'] + " " + result['education_description'],
                    result['education_major'] + " " + result['education_description'] + " " + result['education_activities'],
                    result['education_major'] + " " + result['education_activities'],
                    result['education_description'],
                    result['education_description'] + " " + result['education_activities'],
                    result['education_activities']
                ])
            )

            result['education_school_embedding'] = embeddings[0]
            result['education_degree_embedding'] = embeddings[1]
            result['education_embedding'] = embeddings[2]
            result['education_major_embedding'] = [1, embeddings[9], embeddings[7], embeddings[8], embeddings[3], embeddings[6], embeddings[4], embeddings[5]]

            return result

        # Process each education entry concurrently
        results = await asyncio.gather(*[process_single_education(e) for e in education])
        
        return results

    def process_education(self, target_embedding, similar_embedding):
        
        if not target_embedding or not similar_embedding:
            return 0

        # add the index to each dictionary in the list
        for i in range(len(target_embedding)):
            target_embedding[i]['index'] = i
        for i in range(len(similar_embedding)):
            similar_embedding[i]['index'] = i
        
        target_similar_mapping = [dict() for _ in range(len(target_embedding))] # the index will be the target position index, the content will be the best match index and parts in similar_position_list

        # Generate all combinations of target and similar embeddings
        all_combinations = [(t, s) for t in target_embedding for s in similar_embedding]
        
        combination_similarities = []
        for (t, s) in all_combinations:
            school_similarity = self.calc_school_similarity(t, s)
            degree_similarity = self.calc_degree_similarity(t, s)
            major_similarity = self.calc_major_similarity(t, s)
            time_similarity = self.calc_time_similarity(t, s)
            individual_similarities = [school_similarity, degree_similarity, major_similarity, time_similarity]

            # Combine similarities and store in the list
            similarity_score = school_similarity * degree_similarity * major_similarity * time_similarity
            combination_similarities.append((t, s, similarity_score, individual_similarities))
        
        while combination_similarities:
            # Find the pair with the maximum similarity score
            (t_max, s_max, max_similarity, max_inds) = max(combination_similarities, key=lambda x: x[2])

            # Store the best match in the target_similar_mapping
            target_similar_mapping[t_max['index']] = {
                'match_index': s_max['index'],
                'max_similarity': max_similarity,
                'school_similarity': (t_max['education_school'], s_max['education_school'], max_inds[0]),
                'degree_similarity': (t_max['education_degree'], s_max['education_degree'], max_inds[1]),
                'major_similarity': (t_max['education_major'], s_max['education_major'], max_inds[2]),
                'time_similarity': (t_max['education_startyear'], t_max['education_endyear'], s_max['education_startyear'], s_max['education_endyear'], max_inds[3])
            }

            # Remove combinations involving the matched target and similar entries
            combination_similarities = [
                (t, s, sim, inds) for t, s, sim, inds in combination_similarities 
                if t != t_max and s != s_max
            ]

        return target_similar_mapping

    async def calculate_similarity_weighted(self, target_education, similar_education):
        
        target_embedding, similar_embedding = await asyncio.gather(
                self.get_weighted_embedding(target_education),
                self.get_weighted_embedding(similar_education)
            )

        return self.process_education(target_embedding, similar_embedding)

    async def get_position_info(self, position):
        async def process_single_position(self, pos):
            result = {
                'position_company_name': pos['position_company_name'] if pos['position_company_name'] else ' ',
                'position_end_date': 0,
                'position_location': pos['position_location'] if pos['position_location'] else ' ',
                'position_start_date': 0,
                'position_summary': pos['position_summary'] if pos['position_summary'] else ' ',
                'position_title': pos['position_title'] if pos['position_title'] else ' ',
            }

            if pos['position_end_date'] == 'present' or pos['position_end_date'] is None:
                result['position_end_date'] = datetime.now().date()
            elif pos['position_end_date'] == '':
                result['position_end_date'] = datetime.min.date()
            else:
                result['position_end_date'] = datetime.strptime(pos['position_end_date'], "%b %Y").date()
            
            if pos['position_start_date'] == '':
                result['position_start_date'] = datetime.min.date()
            elif pos['position_start_date'] is None:
                result['position_start_date'] = datetime.min.date()
            else:
                result['position_start_date'] = datetime.strptime(pos['position_start_date'], "%b %Y").date()
            
            embedding = await (
                self.get_embedding([
                    result['position_company_name'], 
                    result['position_location'], 
                    result['position_summary'], 
                    result['position_title']
                ])
            )

            result['position_company_name_embedding'] = embedding[0]
            result['position_location_embedding'] = embedding[1]
            result['position_summary_embedding'] = embedding[2]
            result['position_title_embedding'] = embedding[3]

            return result
        
        results = await asyncio.gather(*[process_single_position(pos) for pos in position])
        return results

    def calc_company_name_similarity(self, target, similar):
        if target['position_company_name'].strip() == '' or similar['position_company_name'].strip() == '':
            return 1
        return np.dot(target['position_company_name_embedding'], similar['position_company_name_embedding'])

    def calc_location_similarity(self, target, similar):
        if target['position_location'].strip() == '' or similar['position_location'].strip() == '':
            return 1
        return np.dot(target['position_location_embedding'], similar['position_location_embedding'])

    def calc_summary_similarity(self, target, similar):
        if target['position_summary'].strip() == '' or similar['position_summary'].strip() == '':
            return 1
        if target['position_summary'] in similar['position_summary'] or similar['position_summary'] in target['position_summary']:
            return 1
        return np.dot(target['position_summary_embedding'], similar['position_summary_embedding'])

    def calc_title_similarity(self, target, similar):
        if target['position_title'].strip() == '' or similar['position_title'].strip() == '':
            return 1
        return np.dot(target['position_title_embedding'], similar['position_title_embedding'])

    def calc_pos_time_similarity(self, target, similar):
        required_fields = ['position_start_date', 'position_end_date']

        for field in required_fields:
            if field not in target or field not in similar:
                return 0
            
        if target['position_start_date'] == ' ' or similar['position_start_date'] == ' ':
            if target['position_end_date'] == ' ' or similar['position_end_date'] == ' ':
                return 0
            if target['position_end_date'] == similar['position_end_date']:
                return 0.7
            if abs((target['position_end_date'] - similar['position_end_date'])) <= timedelta(days = 90):
                return 0.5
            return 0
        elif target['position_end_date'] == ' ' or similar['position_end_date'] == ' ':
            if target['position_start_date'] == similar['position_start_date']:
                return 0.7
            if abs((target['position_start_date'] - similar['position_start_date'])) <= timedelta(days = 90):
                return 0.5
            return 0
        
        # Make sure at this point all time stamps are available
        if target['position_start_date'] <= similar['position_start_date']:
            if target['position_end_date'] >= similar['position_end_date']:
                return 1
            elif (similar['position_end_date'] - target['position_end_date']) <= timedelta(days = 90):
                return 0.7
        else:
            if similar['position_end_date'] >= target['position_end_date']:
                return 1
            elif (target['position_end_date'] - similar['position_end_date']) <= timedelta(days = 90):
                return 0.7
        return 0

    def get_custom_judge(self, target, similar):
        score = int(self.calc_company_name_similarity(target, similar) >= 0.67) + int(self.calc_location_similarity(target, similar) >= 0.75) + int(self.calc_summary_similarity(target, similar) >= 0.7) + int(self.calc_title_similarity(target, similar) >= 0.68)
        return score >= 3

    def process_position(self, target_position_list, similar_position_list):
        
        if not target_position_list or not similar_position_list:
            return 0
        
        # add the index to each dictionary in the list
        for i in range(len(target_position_list)):
            target_position_list[i]['index'] = i
        for i in range(len(similar_position_list)):
            similar_position_list[i]['index'] = i
        
        target_similar_mapping = [dict() for _ in range(len(target_position_list))] # the index will be the target position index, the content will be the best match index and parts in similar_position_list

        # target_similar_mapping = {target_position['position_title']: -1 for target_position in target_position_list}

        threshold = min(math.ceil(0.5 * len(similar_position_list)), math.ceil(len(target_position_list) * 0.5))
        
        all_combinations = [(t, s) for t in target_position_list for s in similar_position_list]
        
        def calculate_combination_similarity(t, s):
            similarities = [
                self.calc_company_name_similarity(t, s),
                self.calc_location_similarity(t, s),
                self.calc_summary_similarity(t, s),
                self.calc_title_similarity(t, s),
                self.calc_pos_time_similarity(t, s)
            ]
            return t, s, np.sum(similarities), similarities
        
        combination_similarities = [calculate_combination_similarity(t, s) for t, s in all_combinations]
        
        while combination_similarities:
            (t_max, s_max, similarity, individual_similarities) = max(combination_similarities, key=lambda x: x[2])

            combination_similarities = [(t, s, sim, inds) for t, s, sim, inds in combination_similarities if t != t_max and s != s_max]

            company_similarity = individual_similarities[0]
            location_similarity = individual_similarities[1]
            summary_similarity = individual_similarities[2]
            title_similarity = individual_similarities[3]
            time_similarity = individual_similarities[4]

            target_similar_mapping[t_max['index']] = {
                'match_index': s_max['index'],
                'total_similarity': similarity,
                'company_similarity': (t_max['position_company_name'], s_max['position_company_name'], company_similarity),
                'location_similarity': (t_max['position_location'], s_max['position_location'], location_similarity),
                'summary_similarity': (t_max['position_summary'], s_max['position_summary'], summary_similarity),
                'title_similarity': (t_max['position_title'], s_max['position_title'], title_similarity),
                'time_similarity': (t_max['position_start_date'].strftime("%Y-%m-%d"), t_max['position_end_date'].strftime("%Y-%m-%d"), s_max['position_start_date'].strftime("%Y-%m-%d"), s_max['position_end_date'].strftime("%Y-%m-%d"), time_similarity)
            }

        print("****************************************** next target similar pair ***************************************************************")

        return target_similar_mapping

    async def calculate_position_all_comb(self, target, similar):

        target_position_list, similar_position_list = await asyncio.gather(
            self.get_position_info(target), 
            self.get_position_info(similar)
        )

        return self.process_position(target_position_list, similar_position_list)

    def get_additional_education_info(self, e):
        education_info = {
            'education_school': e['education_school'] if e['education_school'] != "" else " ", 
            'education_degree': e['education_degree_level'] if e['education_degree_level'] != "" else self.degree_normalization(e['education_degree']), 
            'education_major': e['education_major'] if e['education_major'] != "" else " ", 
            'education_description': e['education_description'] if e['education_description'] != "" else " ",
            'education_activities': e['education_activities'] if e['education_activities'] != "" else " ",
            'education_startyear': e['education_startyear'], 
            'education_endyear': e['education_endyear'],
            'education_string': self.string_from_education([e])
        }
        return education_info

    def get_additional_position_info(self, pos):
        result = {
                'position_company_name': pos['position_company_name'] if pos['position_company_name'] else ' ',
                'position_end_date': 0,
                'position_location': pos['position_location'] if pos['position_location'] else ' ',
                'position_start_date': 0,
                'position_summary': pos['position_summary'] if pos['position_summary'] else ' ',
                'position_title': pos['position_title'] if pos['position_title'] else ' ',
            }

        if pos['position_end_date'] == 'present' or pos['position_end_date'] is None:
            result['position_end_date'] = datetime.now().date()
        elif pos['position_end_date'] == '':
            result['position_end_date'] = datetime.min.date()
        else:
            result['position_end_date'] = datetime.strptime(pos['position_end_date'], "%b %Y").date()
        
        if pos['position_start_date'] == '':
            result['position_start_date'] = datetime.min.date()
        elif pos['position_start_date'] is None:
            result['position_start_date'] = datetime.min.date()
        else:
            result['position_start_date'] = datetime.strptime(pos['position_start_date'], "%b %Y").date()
        return result

    async def get_profile_embeddings(self, profile):
        education = []
        position = []
        transit = []
        data_list = []
        num_education = len(profile['education']) # each education will have 10 embeddings
        
        for e in profile['education']:
            data_list.append(e['education_school'])
            data_list.append(e['education_degree'])
            data_list.append(e['education_school'] + " " + e['education_degree'] + " " + e['education_major'])
            data_list.append(e['education_major'])
            data_list.append(e['education_major'] + " " + e['education_description'])
            data_list.append(e['education_major'] + " " + e['education_description'] + " " + e['education_activities'])
            data_list.append(e['education_major'] + " " + e['education_activities'])
            data_list.append(e['education_description'])
            data_list.append(e['education_description'] + " " + e['education_activities'])
            data_list.append(e['education_activities'])

            # initialize other information needed in the results
            education_info = self.get_additional_education_info(e)
            transit.append(education_info)

        num_position = len(profile['position']) # each position will have 4 embeddings
        for p in profile['position']:
            data_list.append(p['position_company_name'])
            data_list.append(p['position_location'])
            data_list.append(p['position_summary'])
            data_list.append(p['position_title'])

            # initialize other information needed in the results
            position_info = self.get_additional_position_info(p)
            transit.append(position_info)
        
        # remove empty strings
        data_list = [d if d else " " for d in data_list]
        embeddings = await self.get_embedding(data_list)
        start = time.time()
        for _ in range(num_education):
            result = {
                'education_school_embedding' : embeddings[0],
                'education_degree_embedding' : embeddings[1],
                'education_embedding' : embeddings[2],
                'education_major_embedding' : [1, embeddings[9], embeddings[7], embeddings[8], embeddings[3], embeddings[6], embeddings[4], embeddings[5]],
            }
            result.update(transit.pop(0))
            education.append(result)
            embeddings = embeddings[10:]
        for _ in range(num_position):
            result = {
                'position_company_name_embedding' : embeddings[0],
                'position_location_embedding' : embeddings[1],
                'position_summary_embedding' : embeddings[2],
                'position_title_embedding' : embeddings[3],
            }
            result.update(transit.pop(0))
            position.append(result)
            embeddings = embeddings[4:]
        end = time.time()
        print(f"Time taken for finishing up: {end - start}")
        return education, position

    async def calculate_profile_similarity(self, profile1, profile2):
        (education1, position1), (education2, position2) = await asyncio.gather(self.get_profile_embeddings(profile1), self.get_profile_embeddings(profile2))
        education_mapping = self.process_education(education1, education2)
        position_mapping = self.process_position(position1, position2)
        result = {
            'education_mapping': education_mapping,
            'position_mapping': position_mapping
        }

        return result

    async def process(self, data, max_concurrent = 50):
        semaphore = asyncio.Semaphore(max_concurrent)

        async def sem_task(task):
            async with semaphore:
                return await task
            
        flattened_data_profile = [(i['target'], j) for i in data for j in i['similar'].values()]   
        tasks_target_profile = [sem_task(self.calculate_profile_similarity(profile[0], profile[1])) for profile in flattened_data_profile]
        results = await asyncio.gather(*tasks_target_profile)

        return results

async def test():
    calculator = ProfileSimilarityCalculator(api_key)
    semaphore = asyncio.Semaphore(50)
    async def sem_task(task):
        async with semaphore:
            return await task
    print(len(data))
    await asyncio.gather(*[(sem_task(calculator.get_profile_embeddings(profile['target']))) for profile in data])  

async def test2():
    calculator = ProfileSimilarityCalculator(api_key)
    results = await calculator.calculate_profile_similarity(data[0]['target'], list(data[0]['similar'].values())[0])
    print(results)
    return 0

if __name__ == "__main__":
    start = time.time()
    psc = ProfileSimilarityCalculator(api_key)
    with open('export_profiles.jsonl', 'r') as f:
        data = [(json.loads(line)) for line in f]
    result = asyncio.run(psc.process(data, 70))
    print(result)
    end = time.time()
    print(f"Total start time: {start}, Total end time: {end}")
    print(f"Total time taken: {end - start}")
