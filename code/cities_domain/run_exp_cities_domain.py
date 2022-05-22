

import os, json

import pandas as pd

import itertools

import datetime

import random

import openai

import math


def validate_city_pair(city_A, city_B, attribute):
    value_A = df_data[df_data['city'] == city_A][attribute].values[0]
    value_B = df_data[df_data['city'] == city_B][attribute].values[0]
    if attribute == 'avg_temp':
        value_A = float(value_A.split('(')[0].replace('−','-'))
        value_B = float(value_B.split('(')[0].replace('−','-'))
        if (value_A - value_B) >= config['temp_diff']:
            return (city_A, city_B)
        if (value_A - value_B) <= config['temp_diff']*(-1):
            return (city_B, city_A)
    else:
        if (value_A - value_B) >= config['pop_diff']:
            return (city_A, city_B)
        if (value_A - value_B) <= config['pop_diff']*(-1):
            return (city_B, city_A)
    return False


def generate_statements(template_format, pairs):
    template = template_format['template']
    generated = []
    for pair in pairs:
        statement = template.replace('<arg0>', pair[0]).replace('<arg1>', pair[1])
        answer = pair[template_format['answer']]
        candidate = pair[template_format['candidate']]
        statement = randomly_flip_order({'statement' : statement, 'answer' : answer, 'candidate' : candidate})
        generated.append({'statement' : statement, 'answer' : answer, 'candidate' : candidate})
    return generated


def is_answer_before_candidate(answer, wrong, top_predictions):
    idx_wrong = 5
    for i, key in enumerate(top_predictions):
        if wrong.startswith(key):
            idx_wrong = i
            break
    
    idx_answer = 5
    for i, key in enumerate(top_predictions):
        if answer.startswith(key):
            idx_answer = i
            break
    
    if idx_answer < idx_wrong:
        return True
    return False


def randomly_flip_order(example):
    random_decision = random.choice([0, 1])
    if random_decision:
        statement = example['statement']
        answer = example['answer']
        wrong = example['candidate']

        statement = statement.replace(answer, 'BUFFERCITY', 1)
        statement = statement.replace(wrong, answer, 1)
        return statement.replace('BUFFERCITY', wrong, 1)
    return example['statement']


def generate_examples(examples_set, shots, rule_out):
    examples = []
    for example in examples_set:
        if any(city in example['statement'] for city in rule_out):
            continue

        flipped = randomly_flip_order(example)
        examples.append(f"-> {example['answer']}".join(flipped.rsplit(example['answer'], 1)))
        if len(examples) == shots:
            break

    return ' '.join([example for example in examples])


def test_GPT3(test_cases, shots=0):
    tests = []
    for case in test_cases:
        statement = case['statement']
        answer = case['answer']
        wrong = case['candidate']
        prompt = f"{answer.join(statement.split(answer)[:-1]).strip()} ->"

        if shots:
            prompt = f"{generate_examples(test_cases, shots, [answer, wrong])} {prompt}"

        response = openai.Completion.create(
            engine=config['model'], #originally model=
            prompt=prompt,
            temperature=0,
            max_tokens=2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=5
        )

        top_logits = response['choices'][0]['logprobs']['top_logprobs'][0].to_dict()
        top_percents = { key[1:] : math.exp(top_logits[key]) for key in top_logits }
        top_predictions = sorted(top_percents, key=top_percents.get, reverse=True)
        
        tests.append({'prompt' : prompt, 'answer' : answer, 'candidate' : wrong, 'predictions' : top_percents,\
                      'top_prediction' : top_predictions[0],\
                      'predicts_answer_before_candidate' : is_answer_before_candidate(answer, wrong, top_predictions)
                     })
    return tests


config_file = 'exp_config_cities_domain.json'

config = None

with open(os.path.join(os.getcwd(), config_file), 'r') as f_config:
    config = json.load(f_config)

if not config:
    print(f"Couldn't load config file: {config_file}")
    exit()


openai.api_key = config['openai_key']


df_data = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'testing', 'cities_domain', 'test_data_cities.csv'), index_col=0)
cities = list(df_data['city'].values)
print('Facts table loaded')

temp_comparisons = [validate_city_pair(pair[0], pair[1], 'avg_temp') for pair in itertools.combinations(cities, 2)\
                    if validate_city_pair(pair[0], pair[1], 'avg_temp')]
print(f"Generated {len(temp_comparisons)} pairs of cities for temperature comparison")

pop_comparisons = [validate_city_pair(pair[0], pair[1], 'pop') for pair in itertools.combinations(cities, 2)\
                    if validate_city_pair(pair[0], pair[1], 'pop')]
print(f"Generated {len(pop_comparisons)} pairs of cities for population comparison")

samples = {}
samples['temp'] = random.sample(temp_comparisons, config['sample'])
samples['temp_counter'] = samples['temp'] 
samples['pop'] = random.sample(pop_comparisons, config['sample'])
samples['pop_counter'] = samples['pop']
print(f"Generated samples of N={config['sample']}")


successful_cases = {}
successful_cases['temp'] = {}
successful_cases['temp_counter'] = {}
successful_cases['pop'] = {}
successful_cases['pop_counter'] = {}

failed_cases = {}
failed_cases['temp'] = {}
failed_cases['temp_counter'] = {}
failed_cases['pop'] = {}
failed_cases['pop_counter'] = {}

segmented_cases = {}
segmented_cases['temp'] = {}
segmented_cases['temp_counter'] = {}
segmented_cases['pop'] = {}
segmented_cases['pop_counter'] = {}


comparison_scores = []
decision_scores = []
decision_scores_easy = []
decision_scores_hard = []


exp_namespace = f"{config['exp_name']} at {datetime.datetime.now()}"
os.mkdir(exp_namespace)

for template_name in config['factual']:
    # Select and save sample:
    dimension = template_name.split('_')[0]
    if template_name.endswith('_counter'):
        dimension = dimension + '_counter'
    successful_cases[dimension][template_name] = []
    failed_cases[dimension][template_name] = []
    
    test_cases = generate_statements(config['templates'][template_name], samples[dimension])
    
    os.mkdir(os.path.join(os.getcwd(), exp_namespace, template_name))
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'sample.txt'), 'w') as f_sample:
        for statement in test_cases:
            f_sample.write(f"{statement}\n")
    print(f"Generated {config['sample']} statements with '{dimension}' data following template '{template_name}'")
    
    # Run experiment on GPT3:
    tests = test_GPT3(test_cases, config['shots'])
    
    df_tests = pd.DataFrame(tests)
    df_tests.to_csv(os.path.join(os.getcwd(), exp_namespace, template_name, 'results.csv'))
    
    score = sum(df_tests['predicts_answer_before_candidate'].values) / len(df_tests)
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'score.txt'), 'w') as f_result:
        f_result.write(f"{score}")
    print(f"For '{template_name}', GPT3 scored: {score}")
    comparison_scores.append(score)
    
    for i, test in enumerate(tests):
        if test['predicts_answer_before_candidate']:
            successful_cases[dimension][template_name].append(i)
        else:
            failed_cases[dimension][template_name].append(i)

# Split the data between cases where factual templates ALL succeeded and where they ALL failed
for dimension in successful_cases:
    sets = [set(successful_cases[dimension][template]) for template in successful_cases[dimension]]
    segmented_cases[dimension]['all_successful'] = list(sets[0].intersection(*sets))
    sets = [set(failed_cases[dimension][template]) for template in failed_cases[dimension]]
    segmented_cases[dimension]['all_failed'] = list(sets[0].intersection(*sets))

with open(os.path.join(os.getcwd(), exp_namespace, 'segments.txt'), 'w') as f_segments:
    f_segments.write(f"{segmented_cases}")


for template_name in config['templates']:
    if template_name in config['factual']:
        continue
    
    # Select and save sample:
    dimension = template_name.split('_')[0]
    if template_name.endswith('_counter'):
        dimension = dimension + '_counter'
    test_cases = generate_statements(config['templates'][template_name], samples[dimension])

    os.mkdir(os.path.join(os.getcwd(), exp_namespace, template_name))
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'sample.txt'), 'w') as f_sample:
        for statement in test_cases:
            f_sample.write(f"{statement}\n")
    print(f"Generated {config['sample']} statements with '{dimension}' data following template '{template_name}'")

    # Run experiment on GPT3:
    tests = test_GPT3(test_cases, config['shots'])
    df_tests = pd.DataFrame(tests)
    df_tests.to_csv(os.path.join(os.getcwd(), exp_namespace, template_name, 'results.csv'))
    
    # Score the experiment on the entire data:
    score = sum(df_tests['predicts_answer_before_candidate'].values) / len(df_tests)
    print(f"For '{template_name}', GPT3 scored overall: {score}")
    decision_scores.append(score)
    
    # As well as on the data segments:
    segments = segmented_cases[dimension]
    segmented_scores = {}
    segmented_scores['factual_successful'] = 0
    segmented_scores['factual_failed'] = 0
    
    for i, test in enumerate(tests):
        if i in segments['all_successful']:
            if test['predicts_answer_before_candidate']:
                segmented_scores['factual_successful'] += 1
        if i in segments['all_failed']:
            if test['predicts_answer_before_candidate']:
                segmented_scores['factual_failed'] += 1
    
    factual_successful_N = len(segments['all_successful']) if len(segments['all_successful']) else 1
    factual_successful_score = segmented_scores['factual_successful'] / factual_successful_N
    print(f"On the subset of cases where factual templates ALL succeded (N={factual_successful_N}): {factual_successful_score}")
    decision_scores_easy.append(factual_successful_score)
    
    factual_failed_N = len(segments['all_failed']) if len(segments['all_failed']) else 1
    factual_failed_score = segmented_scores['factual_failed'] / factual_failed_N
    print(f"On the subset of cases where factual templates ALL failed (N={factual_failed_N}): {factual_failed_score}")
    decision_scores_hard.append(factual_failed_score)
    
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'scores.txt'), 'w') as f_result:
        f_result.write(f"Overall: {score}\n")
        f_result.write(f"Subset of cases where factual templates ALL succeeded: {factual_successful_score}\n")
        f_result.write(f"Subset of cases where factual templates ALL failed: {factual_failed_score}")


print("\n")
with open(os.path.join(os.getcwd(), exp_namespace, 'final_scores.txt'), 'w') as f_result:
    series = pd.Series(data=comparison_scores)
    output = f"Score on factual comparisons: {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(f"{output}\n")

    series = pd.Series(data=decision_scores)
    output = f"Total score on decision templates: {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(f"{output}\n")

    series = pd.Series(data=decision_scores_easy)
    output = f"Score on decision templates (EASY): {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(f"{output}\n")

    series = pd.Series(data=decision_scores_hard)
    output = f"Score on decision templates (HARD): {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(output)
print("\n")

