
## "Learning to Perform Complex Tasks through Compositional Fine-Tuning of Language Models." (To appear, Findings of EMNLP 2022)

This repository contains code and data resources for the paper submission "Learning to Perform Complex Tasks through Compositional Fine-Tuning of Language Models," authored by Victor S. Bursztyn (v-bursztyn@u.northwestern.edu), David Demeter, Doug Downey, and Larry Birnbaum.

These resources are organized as follows:

- Inside folder `code`, subfolders `cities_domain`, `restaurants_domain`, and `sports_understanding` contain the code for running experiments for each one of our tasks.
- Inside folder `data`:
	- In subfolder `phrasings`, we have the different natural language patterns used for decision templates and factual comparisons in the recommendation task (both ciies and restaurants) as well as the negative preference interpretations;
	- In subfolder `training`, we have the fine-tuning data for each CFT configuration tested in each experiment, which means that:
		- In subfolder `experiment_1`, we have all ablations seen on Tables 2, 3, 4, and 5;
		- In subfolder `experiment_2`, we have all ablations seen on Tables 6 and 7;
		- In subfolder `experiment_4`, we have the two folds of the sports understanding task (Section 5.5.4);
		- Please note: since Experiment 3 doesn't involve fine-tuning, only chain of thought prompting, it is in a different folder; and fine-tuning data is always in their raw format (subfolders named `raw_fine_tuning`) as well as in JSONL format after preprocessing with OpenAI's command-line utility (subfolders named `prepared_fine_tuning`).
	- In subfolder `chain_of_thought_prompting`, we have the full prompts used in the recommendation task (both cities and restaurants) for Experiment 3;
	- In subfolder `testing`, we have the test data used in the recommendation task (again, both cities and restaurants) as well as in the sports understanding task.
- Inside folder `results`, subfolders `experiment_1`, `experiment_2`, `experiment_3`, and `experiment_4` contain the output of all experimental runs, as follows:
	- In subfolder `experiment_1`, `component_ablation` and `equal_data_amounts` refer to Tables 2-3 and 4-5, respectively;
	- In subfolder `experiment_2`, `cities_domain` and `restaurants_domain` refer to Tables 6 and 7, respectively;
	- In subfolder `experiment_3`, `cities_domain` and `restaurants_domain` cover different rows in Table 1;
	- In subfolder `experiment_4`, we have each run in the 2-fold cross-validation for the sports understanding task;
	- When opening an experimental run, globally averaged scores can be found in `final_scores.txt` at that run's root level, while the output of individual phrasings can be found in the subfolders;
	- At this level, you can also find the run's configuration file named `exp_config_*.json`.

The code and data provided were used to produce these experimental runs. To reproduce them, you need to:
1. Install pandas and openai (e.g., `pip install -r requirements.txt` with the `requirements.txt` provided at the repository's root level -- recommended for versioning consistency);
2. Copy an experimental run's configuration file to the folder of the script that is going to be executed;
3. Edit this configuration file to include a valid OpenAI key (currently defined as `"openai_key" : "INSERT YOUR OPEN AI API KEY HERE"` in all configuration files);
4. And simply execute one of the scripts from `code`.
	- Execution at this point is possible because all configuration files have references to the language models we fine-tuned on OpenAI's cloud using CFT;
	- However, you can also re-run CFT and simply update the parameter `"model" : "(...)"` with the newly fine-tuned model to reproduce the entire process, from CFT to evaluation output;
		- In this case, please use a JSONL file from any `prepared_fine_tuning` subfolder inside `data`, as follows:

		```
		export OPENAI_API_KEY="<YOUR_OPEN_AI_KEY>"
		openai api fine_tunes.create -t <INSERT_JSONL_FILE> -m curie
		```
		- The installation of openai in step #1 includes the [command-line utility](https://beta.openai.com/docs/libraries/python-bindings) mentioned above.
	- Finally, configuration files have the parameter `sample` that determines the sample size -- you can use this parameter to scale experimental costs accordingly.

---
Please cite this paper as: TBD

