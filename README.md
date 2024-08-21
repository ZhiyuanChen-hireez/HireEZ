# ProfileSimilarityCalculator

## Overview
This project is a simple implementation of a profile similarity calculator. It calculates the similarity between two profiles based on the cosine similarity of the word embeddings of the profiles. The project is implemented in Python and uses the openai embeddings api to get the word embeddings.

## Usage
To use the project, you need to have an openai api key. You can get one by signing up at https://beta.openai.com/signup/. Once you have the api key, you can follow a similar code pattern at Usage.ipynb.

## Input data structure
The input data should of the structure:
```
{
    "education": ["education1", "education2", ...],
    "position": ["position1", "position2", ...],
}
```
where the elements of the lists are dictionaries.

## Key ideas
most of the functions are self-explanatory. The key idea is to get the word embeddings of the profiles and then calculate the cosine similarity between the embeddings. There are some hand crafted data structure to optimize the concurency of the code. I will introduce the key ideas in a top-down manner.

### process(self, data, max_concurrent)
This function is the driver function of the class. It takes the input data and the maximum number of concurrent requests to the openai api. It then processes the data and returns the similarity matrix. Note it uses the 'calculate_profile_similarity' function to calculate the similarity between two profiles.

### calculate_profile_similarity(self, profile1, profile2)
This function mainly collects all information needed and put them in a dictionary. Note it calls the 'get_profile_embeddings', 'process_education' and 'process_position' functions to get the embeddings of the profiles.

### get_profile_embeddings(self, profile)
For concurrency reasons, this function collects all the words in the profile and then calls the 'get_embeddings' function once to get the embeddings of the words. We then match the words with their embeddings according to the index of embedding in the returned list. So order is important here. Pay particular attention if you want to change the implementation of this function.

### process_position(self, target_position_list, similar_position_list)
This function process the position embeddings to find the best matches between target and similar embeddings. The key idea here is to find the best match for each target position embedding. To do this, We calculate the relavant similarity scores for all the target position and similar position pairs (check all_combinations and combination_similarity variables) and then find the best match pairs until all the target position embeddings or similar position embeddings are matched. (check the while loop).

### process_education(self, target_education_list, similar_education_list)
This function is similar to the process_position function. The only thing worth noted is the structure of returned 'target_similar_mapping' value.

### get_embeddings(self, text, max_retries, initial_delay, timeout)
This function is a simple wrapper around the openai api. It retries the request if it fails and returns the embeddings of the words in the text. Thanks to openai, we can pass multiple words in a list in a single request to get the embeddings of all of them. This is the key idea behind the 'get_profile_embeddings' function.

### Other functions
You might already noticed that some functions are not used in the driver function. They are deprecated and not used in the current implementation. They are kept for future reference. I did write some annotations to explain the purpose of each function as well. You can check them if you want to know more about the implementation.