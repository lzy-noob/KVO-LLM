
# * Supported tasks for LM Evaluation Harness & LongBench 
TASKS = {
    "longbench": {
        'test_set': ['triviaqa'],
        'quest_set': ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa","gov_report", "triviaqa"], \
        'single_doc': ["multifieldqa_en","narrativeqa","qasper"], \
        'multi_doc': ["hotpotqa","2wikimqa","musique"], \
        'summary': ["gov_report", "qmsum", "multi_news"], \
        'few_shot': ["triviaqa","trec"], \
        'synthetic': ["passage_retrieval_en", "passage_count"], \
        'code': ["lcc", "repobench-p"], \
        'full_set': ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    }
}

# * LongBench
LONGBENCH_PATH = "longbench"